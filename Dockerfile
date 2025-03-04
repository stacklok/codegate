# Builder stage: Install dependencies and build the application
FROM docker.io/library/python:3.12-slim@sha256:34656cd90456349040784165b9decccbcee4de66f3ead0a1168ba893455afd1e AS builder

ARG CODEGATE_VERSION=dev

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.4 && rm -rf /root/.cache/pip

# Set the working directory
WORKDIR /app
COPY pyproject.toml poetry.lock* /app/

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy the rest of the application
COPY . /app

# Overwrite the _VERSION variable in the code
RUN sed -i "s/_VERSION =.*/_VERSION = \"${CODEGATE_VERSION}\"/g" /app/src/codegate/__init__.py

# Build the webapp
FROM docker.io/library/node:23-slim@sha256:dcacc1ee3b03a497c2096b0084d3a67b856e777b55ffccfcc76bcdab9cc65906 AS webbuilder



# Install curl for downloading the webapp from GH and unzip to extract it
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    unzip\
    ca-certificates

WORKDIR /usr/src/

# Set build arg for latest release URL (optional)
ARG LATEST_RELEASE

# Download the latest release - if LATEST_RELEASE is provided use it, otherwise fetch from API
RUN if [ -n "$LATEST_RELEASE" ]; then \
        echo "Using provided release URL" && \
        curl -L -o main.zip "${LATEST_RELEASE}"; \
    else \
        echo "Fetching latest release URL" && \
        curl -s https://api.github.com/repos/stacklok/codegate-ui/releases/latest | \
        jq -r '.zipball_url' | xargs curl -L -o main.zip; \
    fi

# Extract the downloaded zip file
RUN unzip main.zip
RUN rm main.zip
# Rename the extracted folder
RUN mv *codegate-ui* webapp

WORKDIR /usr/src/webapp

# Install the webapp dependencies and build it
RUN npm install
RUN npm run build

# Runtime stage: Create the final lightweight image
FROM python:3.12-slim AS runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 -r codegate


# Set permissions for user codegate to run nginx
RUN chown -R codegate /var/lib/nginx && \
    chown -R codegate /var/log/nginx && \
    chown -R codegate /run

COPY nginx.conf /etc/nginx/nginx.conf

# Remove include /etc/nginx/sites-enabled/*; from the default nginx.conf
# This way we don't introduce unnecessary configurations nor serve
# any default content.
RUN sed -i '/sites-enabled/d' /etc/nginx/nginx.conf

# Switch to codegate user
USER codegate
WORKDIR /app

# Copy necessary artifacts from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Copy necessary artifacts from the webbuilder stage
COPY --from=webbuilder /usr/src/webapp/dist /var/www/html
USER root
RUN chown -R codegate /var/www/html
USER codegate

# Expose nginx
EXPOSE 9090

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app/src

# Expose additional env vars
ENV CODEGATE_VLLM_URL=
ENV CODEGATE_OPENAI_URL=
ENV CODEGATE_ANTHROPIC_URL=
ENV CODEGATE_OLLAMA_URL=http://host.docker.internal:11434
ENV CODEGATE_LM_STUDIO_URL=http://host.docker.internal:1234
ENV CODEGATE_APP_LOG_LEVEL=WARNING
ENV CODEGATE_LOG_FORMAT=TEXT

# Copy the initial models in the image to default models
RUN mkdir -p /app/default_models && cp /app/codegate_volume/models/* /app/default_models/

# Define volume for persistent data
VOLUME ["/app/codegate_volume/"]

# This has to be performed after copying from the builder stages.
# Otherwise, the permissions will be reset to root.
USER root
RUN mkdir -p /app/codegate_volume/db
# Make codegate user the owner of codegate_volume directory to allow writing to it
RUN chown -R codegate /app/codegate_volume
USER codegate

# Set the container's default entrypoint
EXPOSE 8989
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
