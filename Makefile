.PHONY: test docs

install:
	poetry install --with dev --with test
	poetry run pre-commit install --install-hooks

clean:
	rm -rf .memos
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf tmp

test:
	PYTHONPATH=src poetry run pytest tests

format:
	poetry run ruff check --fix
	poetry run ruff format

pre_commit:
	poetry run pre-commit run -a

serve:
	poetry run uvicorn memos.api.start_api:app

docs:
	pytest tests/test_docs.py -xq
	pip install mkdocs mkdocs-material -q
	mkdir -p tmp
	rm -f tmp/mkdocs.yml
	echo "site_name: MemOS Documentation" > tmp/mkdocs.yml
	echo "docs_dir: `pwd`/docs" >> tmp/mkdocs.yml
	echo "theme: material" >> tmp/mkdocs.yml
	echo "markdown_extensions:" >> tmp/mkdocs.yml
	echo "  - pymdownx.highlight:" >> tmp/mkdocs.yml
	echo "      anchor_linenums: true" >> tmp/mkdocs.yml
	echo "      line_spans: __span" >> tmp/mkdocs.yml
	echo "      pygments_lang_class: true" >> tmp/mkdocs.yml
	echo "  - pymdownx.inlinehilite" >> tmp/mkdocs.yml
	echo "  - pymdownx.snippets" >> tmp/mkdocs.yml
	echo "  - pymdownx.superfences" >> tmp/mkdocs.yml
	cat docs/settings.yml >> tmp/mkdocs.yml
	mkdocs serve -f tmp/mkdocs.yml --no-livereload

openapi:
	poetry run python scripts/export_openapi.py --output docs/openapi.json
