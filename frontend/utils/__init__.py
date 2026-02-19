"""
EasyRead Frontend Utilities

This package contains helper modules for:
- backend.py: API communication with the EasyRead backend
- docx_export.py: Export functionality to Microsoft Word format
- markdown_export.py: Export functionality to Markdown format
"""

from .backend import BackendClient
from .docx_export import export_to_docx
from .markdown_export import export_to_markdown

__all__ = ["BackendClient", "export_to_docx", "export_to_markdown"]
