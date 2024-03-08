import os
import importlib

# Get the directory path where the current __init__.py file resides
current_dir = os.path.dirname(__file__)

# Get a list of all Python files in the directory (excluding __init__.py)
module_files = [file[:-3] for file in os.listdir(current_dir) if file.endswith('.py') and file != '__init__.py']

# Import all modules dynamically and add them to the package namespace
for module_name in module_files:
    module = importlib.import_module('.' + module_name, package=__name__)
    globals()[module_name] = module