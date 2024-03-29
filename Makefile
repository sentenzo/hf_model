PROJECT_DIR = hf_model

CODE = ${PROJECT_DIR} tests

run:
	poetry run python -m ${PROJECT_DIR}

init:
	poetry install --no-root

export_requirements:
	poetry export --without-hashes -f requirements.txt --output requirements.txt
	poetry export --without-hashes --with dev -f requirements.txt --output requirements.dev.txt

lint:
	poetry run mypy ${CODE}
	poetry run isort ${CODE}
	poetry run black ${CODE}
	poetry run flake8 ${CODE} --count --show-source --statistics

test:
	poetry run pytest -vsx -m "not slow"

test-all:
	poetry run pytest -vsx