from pathlib import Path

EXCLUDE_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
    "eda",
}

INCLUDE_EXTENSIONS = {
    ".py", ".md", ".json", ".yaml", ".yml",
    ".toml", ".txt", ".sh", ".bat", ".sql"
}

def show(path, prefix=""):
    items = []

    for p in path.iterdir():
        if p.is_dir() and p.name in EXCLUDE_DIRS:
            continue

        if p.is_file() and p.suffix.lower() not in INCLUDE_EXTENSIONS:
            continue

        items.append(p)

    items.sort(key=lambda x: (x.is_file(), x.name.lower()))

    for i, p in enumerate(items):
        last = i == len(items) - 1
        branch = "+-- " if last else "|-- "

        print(prefix + branch + p.name)

        if p.is_dir():
            show(p, prefix + ("    " if last else "|   "))


print(Path(".").resolve().name)
show(Path("."))