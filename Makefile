.PHONY: clean install format lint test security build all
CONTAINER_BUILD?=docker buildx build
# This is the container tag. Only used for development purposes.
VER?=latest

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -f .coverage
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

install:
	poetry install --with dev

format:
	poetry run black .
	poetry run ruff check --fix .

lint:
	poetry run black --check .
	poetry run ruff check .

test:
	poetry run pytest

security:
	poetry run bandit -r src/

build: clean test
	poetry build

image-build:
	DOCKER_BUILDKIT=1 $(CONTAINER_BUILD) \
		-f Dockerfile \
		--build-arg LATEST_RELEASE=$(curl -s "https://api.github.com/repos/stacklok/codegate-ui/releases/latest" | grep '"zipball_url":' | cut -d '"' -f 4) \
		--build-arg CODEGATE_VERSION="$(shell git describe --tags --abbrev=0)-$(shell git rev-parse --short HEAD)-dev" \
		-t codegate \
		. \
		-t ghcr.io/stacklok/codegate:$(VER) \
		--load

all: clean install format lint test security build
