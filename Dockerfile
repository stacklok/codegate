# Builder stage: Install dependencies and build the application
FROM python:3.12-slim AS builder

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

# Build the webapp
FROM node:23-slim AS webbuilder

# Install curl for downloading the webapp from GH and unzip to extract it
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip\
    ca-certificates

WORKDIR /usr/src/

# To ensure we always download the latest release of the webapp, we use a build argument.
# This prevents the curl command from being cached by Docker.
ARG LATEST_RELEASE="v0.0.9"
RUN echo "Latest FE release: $LATEST_RELEASE"
RUN LATEST_RELEASE=${LATEST_RELEASE} \
    curl -L -o main.zip "https://api.github.com/repos/stacklok/codegate-ui/zipball/${LATEST_RELEASE}"

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

# Copy backup if needed
RUN mkdir -p /tmp/weaviate_backup
# will not fail if the file does not exist
COPY weaviate_backu[p] /tmp/weaviate_backup
RUN chown -R codegate /tmp/weaviate_backup

# Set permissions for user codegate to run nginx
RUN chown -R codegate /var/lib/nginx && \
    chown -R codegate /var/log/nginx && \
    chown -R codegate /run

COPY nginx.conf /etc/nginx/conf.d/default.conf

# Switch to codegate user
USER codegate
WORKDIR /app

# Copy necessary artifacts from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Copy necessary artifacts from the webbuilder stage
COPY --from=webbuilder /usr/src/webapp/dist /var/www/html
# Expose nginx
EXPOSE 80

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app/src

# Expose additional env vars
ENV CODEGATE_VLLM_URL=
ENV CODEGATE_OPENAI_URL=
ENV CODEGATE_ANTHROPIC_URL=
ENV CODEGATE_OLLAMA_URL=http://host.docker.internal:11434
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