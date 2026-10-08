from pathlib import Path

# Folders that make up the project skeleton
folders = ["data/raw", "data/processed", "notebooks", "src", "models",
           "results/metrics", "results/figures", "results/error_analysis", "app", "tests"]
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)  # parents=True creates nested folders; exist_ok=True avoids errors on re-run

# Empty starter files
for file in ["README.md", "requirements.txt", ".gitignore", "src/__init__.py"]:
    Path(file).touch()

print("Project structure created.")
