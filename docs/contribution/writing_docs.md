# How to Write and Preview Documentation

This project uses a custom Nuxt frontend to render documentation from Markdown files. The documentation is deployed at [https://memos.openmem.net/docs/home](https://memos.openmem.net/docs/home).

## Adding a New Document

1. Create a new `.md` file in the `docs/` directory or one of its subdirectories.
2. Add content to the file using Markdown syntax.
3. Add the new document to the `nav` section in `docs/settings.yml`.
4. Once the your changes are merged into the `main` branch, the documentation will be automatically updated.

## Navigation Icons

When adding entries to the navigation in `docs/settings.yml`, you can include icons using the syntax `(ri:icon-name)`. For example:

```yaml
- "(ri:home-line) Home": overview.md
- "(ri:team-line) Users": modules/mos/users.md
- "(ri:flask-line) Writing Tests": contribution/writing_tests.md
```

The frontend will render these as actual icons. You can find available icons at [https://icones.js.org/](https://icones.js.org/).

## Previewing the Documentation

To preview a simple version of the documentation locally, run the following command from the root of the project:

```bash
make docs
```

This command will start a local web server, and you can view the documentation by opening the URL provided in the terminal (usually `http://127.0.0.1:8000`).
