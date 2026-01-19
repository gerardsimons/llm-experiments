# Gemini CLI Project Notes

- For all package management operations (installing, upgrading, uninstalling dependencies), always use `uv`. This ensures consistency with the project's dependency locking and virtual environment management. For example: `uv pip install <package_name>`.
- To run Python scripts or modules within the project's virtual environment, always use `uv run python <script_name.py>`. This ensures the script executes with the correct interpreter and has access to all installed project dependencies.