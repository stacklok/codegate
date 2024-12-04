# Builder stage: Install dependencies and build the application
FROM python:3.13-slim AS builder

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

# Runtime stage: Create the final lightweight image
FROM python:3.13-slim AS runtime

# Accept build argument for weaviate_data path
ARG WEAVIATE_DATA_PATH

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --system --no-create-home codegate --uid 1000
USER codegate
WORKDIR /app

# Copy necessary artifacts from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app/src

# Copy weaviate_data into the container if the build argument exists
# This step will only work if the build argument is set
RUN if [ -n "$WEAVIATE_DATA_PATH" ]; then \
        echo "Copying weaviate_data from $WEAVIATE_DATA_PATH"; \
        cp -r $WEAVIATE_DATA_PATH /app/weaviate_data; \
    else \
        echo "No WEAVIATE_DATA_PATH provided, skipping copy."; \
    fi

# Allow to expose weaviate_data volume
VOLUME ["/app/weaviate_data"]

# Set the container's default entrypoint
EXPOSE 8989
ENTRYPOINT ["python", "-m", "src.codegate.cli", "serve", "--port", "8989", "--host", "0.0.0.0"]
