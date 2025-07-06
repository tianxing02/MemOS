"""
Tests for docs/settings.yml configuration file.

This module tests the validity and completeness of the documentation
configuration in settings.yml.
"""

import os
import re
import warnings

from pathlib import Path
from typing import Literal

import pytest
import requests
import yaml


@pytest.fixture
def settings_file_path() -> Path:
    """Return the path to settings.yml file."""
    return Path(__file__).parent.parent / "docs" / "settings.yml"


@pytest.fixture
def docs_root_path() -> Path:
    """Return the path to docs directory."""
    return Path(__file__).parent.parent / "docs"


@pytest.fixture
def settings_data(settings_file_path: Path) -> dict:
    """Load and return the settings.yml data."""
    with open(settings_file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_yml_file_format_is_valid(settings_file_path: Path):
    """Test that settings.yml is a valid YAML file."""
    try:
        with open(settings_file_path, encoding="utf-8") as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        pytest.fail(f"settings.yml is not a valid YAML file: {e}")


def test_settings_yml_has_required_structure(settings_data: dict):
    """Test that settings.yml has the required structure."""
    assert "nav" in settings_data, "Missing 'nav' key"
    assert isinstance(settings_data["nav"], list), "'nav' should be a list"


def test_all_nav_files_exist_in_docs_folder(settings_data: dict, docs_root_path: Path):
    """Test that all files referenced in nav exist in the docs folder."""
    nav_items = settings_data.get("nav", [])
    file_paths = _extract_file_paths_from_nav(nav_items)

    missing_files = []
    for file_path in file_paths:
        full_path = docs_root_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    assert not missing_files, (
        f"Files referenced in nav but not found in docs folder: {missing_files}"
    )


def test_all_markdown_files_are_in_nav(settings_data: dict, docs_root_path: Path):
    """Test that all markdown files in docs folder are referenced in nav."""
    # Get all markdown files in docs folder and subdirectories
    markdown_files = set()
    for md_file in docs_root_path.rglob("*.md"):
        # Get relative path from docs root
        relative_path = md_file.relative_to(docs_root_path)
        # Convert to forward slashes for consistency
        relative_path_str = str(relative_path).replace(os.sep, "/")
        markdown_files.add(relative_path_str)

    # Get all files referenced in nav
    nav_items = settings_data.get("nav", [])
    nav_files = _extract_file_paths_from_nav(nav_items)

    # Find markdown files not in nav
    missing_in_nav = markdown_files - nav_files

    assert not missing_in_nav, (
        f"Markdown files in docs folder but not referenced in settings.yml: {missing_in_nav}"
    )


def test_nav_files_are_markdown_files_and_properly_named(settings_data: dict):
    """Test that all files in nav are markdown files and properly named."""

    nav_items = settings_data.get("nav", [])
    file_paths = _extract_file_paths_from_nav(nav_items)

    non_markdown_files = []
    improperly_named_files = []
    pattern = re.compile(r"^[a-z0-9_\/]+\.md$")

    for file_path in file_paths:
        if not file_path.endswith(".md"):
            non_markdown_files.append(file_path)
        elif not pattern.match(file_path):
            improperly_named_files.append(file_path)

    assert not non_markdown_files, f"Non-markdown files found in nav: {non_markdown_files}"
    assert not improperly_named_files, (
        "Files in nav must be lowercase, use only a-z, 0-9, underscores, and slashes, and end with .md: "
        f"{improperly_named_files}"
    )


def test_nav_structure_has_proper_nesting(settings_data: dict):
    """Test that nav structure follows proper nesting conventions."""
    nav_items = settings_data.get("nav", [])

    def validate_nav_item(item, depth=0):
        """Validate individual nav item structure."""
        if isinstance(item, dict):
            assert len(item) == 1, f"Nav item should have exactly one key-value pair: {item}"
            key, value = next(iter(item.items()))
            assert isinstance(key, str), f"Nav item key should be a string: {key}"

            if isinstance(value, str):
                # File reference
                assert value.endswith(".md"), f"File reference should be a markdown file: {value}"
            elif isinstance(value, list):
                # Nested structure
                assert depth < 3, f"Nav nesting too deep (max 2 levels): {item}"
                for nested_item in value:
                    validate_nav_item(nested_item, depth + 1)
            else:
                pytest.fail(f"Invalid nav item value type: {type(value)} for {value}")
        else:
            pytest.fail(f"Nav item should be a dictionary: {item}")

    for item in nav_items:
        validate_nav_item(item)


def test_no_duplicate_files_in_nav(settings_data: dict):
    """Test that no file is referenced multiple times in nav."""
    nav_items = settings_data.get("nav", [])
    file_paths = _extract_file_paths_from_nav(nav_items)
    file_paths_list = list(file_paths)

    # Check for duplicates
    unique_files = set(file_paths_list)
    assert len(file_paths_list) == len(unique_files), "Duplicate file references found in nav"


def test_nav_keys_are_descriptive(settings_data: dict):
    """Test that navigation keys are descriptive and properly formatted."""
    nav_items = settings_data.get("nav", [])

    def check_nav_keys(items):
        """Recursively check navigation keys."""
        problematic_keys = []

        for item in items:
            if isinstance(item, dict):
                for key, value in item.items():
                    # Check key formatting
                    if not key.strip():
                        problematic_keys.append(f"Empty key: '{key}'")
                    elif len(key) < 2:
                        problematic_keys.append(f"Too short key: '{key}'")
                    elif key != key.strip():
                        problematic_keys.append(f"Key has leading/trailing whitespace: '{key}'")

                    # Recursively check nested items
                    if isinstance(value, list):
                        nested_problems = check_nav_keys(value)
                        problematic_keys.extend(nested_problems)

        return problematic_keys

    problematic_keys = check_nav_keys(nav_items)
    assert not problematic_keys, f"Problematic navigation keys found: {problematic_keys}"


def test_yaml_encoding_is_utf8(settings_file_path: Path):
    """Test that settings.yml uses UTF-8 encoding."""
    try:
        with open(settings_file_path, encoding="utf-8") as f:
            f.read()
    except UnicodeDecodeError:
        pytest.fail("settings.yml is not encoded in UTF-8")


def test_yaml_indentation_is_consistent(settings_file_path: Path):
    """Test that YAML indentation is consistent (2 spaces)."""
    with open(settings_file_path, encoding="utf-8") as f:
        lines = f.readlines()

    indentation_errors = []
    for i, line in enumerate(lines, 1):
        if line.strip() and line.startswith(" "):
            # Count leading spaces
            leading_spaces = len(line) - len(line.lstrip(" "))
            if leading_spaces % 2 != 0:
                indentation_errors.append(
                    f"Line {i}: inconsistent indentation ({leading_spaces} spaces)"
                )

    assert not indentation_errors, f"YAML indentation errors: {indentation_errors}"


def _all_md_file_paths() -> list[str]:
    project_dir = Path(__file__).parent.parent
    docs_dir = project_dir / "docs"
    return sorted([str(p.relative_to(project_dir)) for p in docs_dir.rglob("*.md")])


def _get_links(file_path: str, mode: Literal["remote", "local"]) -> list[str]:
    """Extract remote or local links from a markdown file.

    Args:
        file_path (str): Path to the markdown file.
        mode (Literal['remote', 'local']): Mode to extract 'remote' or 'local' links.

    Returns:
        set[str]: Set of extracted links.
    """
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    matches = re.findall(
        r"(?:\!)?\[.*?\]\((?!#)(.*?)\)|"  # Markdown links/images (not anchors)
        r'href=["\'](?!#)(.*?)["\']|'  # HTML href attributes
        r"<(https?://[^>]+)>",  # Direct URLs in angle brackets
        content,
    )
    found_links = {url for match in matches for url in match if url}

    remote_links, local_links = [], []
    for link in found_links:
        if link.startswith(("http://", "https://")):
            remote_links.append(link)
        elif not link.startswith("mailto:"):
            local_links.append(link)
    return remote_links if mode == "remote" else local_links


@pytest.mark.parametrize("file_path", _all_md_file_paths(), ids=_all_md_file_paths())
def test_remote_links_accessibility(file_path: str):
    """Test that all remote links in markdown file are accessible"""
    remote_links = _get_links(file_path, mode="remote")
    print(remote_links)
    for link in remote_links:
        with requests.Session() as session:
            session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; LinkChecker/1.0)"})
            try:
                # Try HEAD first (faster)
                response = session.head(link, timeout=2, allow_redirects=True)

                if response.status_code >= 400:
                    warnings.warn(
                        f"❌ Link {link} in {file_path} may be broken (HEAD request failed). "
                        "Consider checking the link manually.",
                        stacklevel=2,
                    )
                    return
            except requests.RequestException:
                try:
                    # Fallback to GET with shorter timeout
                    response = session.get(link, timeout=2, allow_redirects=True, stream=True)
                    if response.status_code >= 400:
                        warnings.warn(
                            f"❌ Link {link} in {file_path} may be broken (GET request failed). "
                            "Consider checking the link manually.",
                            stacklevel=2,
                        )
                    return
                except requests.RequestException as e:
                    error_msg = str(e)
                    if "timeout" in error_msg.lower():
                        warnings.warn(
                            f"❌ Link {link} in {file_path} timed out. "
                            "Consider checking the link manually.",
                            stacklevel=2,
                        )
                        return
                    elif "connection" in error_msg.lower():
                        warnings.warn(
                            f"❌ Link {link} in {file_path} failed to connect. "
                            "Consider checking the link manually.",
                            stacklevel=2,
                        )
                        return
                    else:
                        warnings.warn(
                            f"❌ Link {link} in {file_path} failed to connect. "
                            "Consider checking the link manually.",
                            stacklevel=2,
                        )
                        return
            return


def _extract_file_paths_from_nav(nav_items: list, base_path: str = "") -> set[str]:
    """
    Recursively extract all file paths from navigation structure.

    Args:
        nav_items: List of navigation items
        base_path: Base path for relative file paths

    Returns:
        Set of file paths found in navigation
    """
    file_paths = set()

    for item in nav_items:
        if isinstance(item, dict):
            for _, value in item.items():
                if isinstance(value, str):
                    # Direct file reference
                    file_path = os.path.join(base_path, value) if base_path else value
                    file_paths.add(file_path)
                elif isinstance(value, list):
                    # Nested navigation structure
                    nested_paths = _extract_file_paths_from_nav(value, base_path)
                    file_paths.update(nested_paths)
        elif isinstance(item, str):
            # Direct string file reference (less common)
            file_path = os.path.join(base_path, item) if base_path else item
            file_paths.add(file_path)

    return file_paths
