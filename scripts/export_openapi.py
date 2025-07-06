import argparse
import json

from memos.api.start_api import app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OpenAPI schema to JSON file.")
    parser.add_argument(
        "--output", type=str, default="docs/openapi.json", help="Output path for OpenAPI schema."
    )
    args = parser.parse_args()
    with open(args.output, "w") as f:
        json.dump(app.openapi(), f, indent=2)
        f.write("\n")
    print("Export completed successfully")
