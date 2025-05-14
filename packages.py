import sys
import importlib
import pkgutil

original_modules = set(sys.modules.keys())

# Run your script here
exec(open("huggingface.py").read())

# Check for newly imported modules
new_modules = set(sys.modules.keys()) - original_modules
imported_packages = [mod for mod in new_modules if not mod.startswith('_')]

print("Imported packages:", imported_packages)
