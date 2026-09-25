DOCKER ?= docker
IMAGE ?= probabilistic-reflex-suppressor-dev:local
CONTAINER_WORKDIR ?= /workspace
DOCKER_RUN_FLAGS ?=
INPUT ?= input/test.jpg
ANNOTATED_OUTPUT ?= output/annotated.png

DOCKER_RUN = $(DOCKER) run --rm --init $(DOCKER_RUN_FLAGS) \
	--env PYTHONDONTWRITEBYTECODE=1 \
	--env PYTHONUNBUFFERED=1 \
	--env RUFF_CACHE_DIR=/tmp/ruff-cache \
	--env MYPY_CACHE_DIR=/tmp/mypy-cache \
	--env COVERAGE_FILE=/tmp/.coverage \
	--volume "$(CURDIR):$(CONTAINER_WORKDIR)" \
	--workdir "$(CONTAINER_WORKDIR)" \
	$(IMAGE)

PYTHON_IN_CONTAINER ?= python

.PHONY: install docker-build format format-check lint typecheck test check run annotate clean

install: docker-build

docker-build:
	$(DOCKER) build --target development --tag $(IMAGE) .

format: docker-build
	$(DOCKER_RUN) $(PYTHON_IN_CONTAINER) -m ruff format src tests

format-check: docker-build
	$(DOCKER_RUN) $(PYTHON_IN_CONTAINER) -m ruff format --check src tests

lint: docker-build
	$(DOCKER_RUN) $(PYTHON_IN_CONTAINER) -m ruff check src tests

typecheck: docker-build
	$(DOCKER_RUN) $(PYTHON_IN_CONTAINER) -m mypy src

test: docker-build
	$(DOCKER_RUN) $(PYTHON_IN_CONTAINER) -m pytest -p no:cacheprovider

check: format-check lint typecheck test

run: docker-build
	$(DOCKER_RUN) $(PYTHON_IN_CONTAINER) -m probabilistic_reflex_suppressor $(ARGS)

annotate: docker-build
	$(DOCKER_RUN) $(PYTHON_IN_CONTAINER) -m probabilistic_reflex_suppressor $(INPUT) --annotated-output $(ANNOTATED_OUTPUT) $(if $(OUTPUT),--output $(OUTPUT))

clean: docker-build
	$(DOCKER_RUN) sh -c "find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' -o -name '.ruff_cache' \) -prune -exec rm -rf {} +; rm -f .coverage"
