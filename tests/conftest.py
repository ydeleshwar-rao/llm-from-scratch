"""Add the project root to sys.path so tests can import all packages."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
