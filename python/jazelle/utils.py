"""
General utility functions and decorators for the Jazelle package.
"""

import functools
import importlib.util
import shutil
from packaging import version
from typing import Dict, Optional, Callable, Any

import numpy as np

from .config import display as global_display

def requires_packages(packages: Optional[Dict[str, str]] = None) -> Callable:
    """
    Decorator to enforce optional dependencies for functions or classes.
    
    The check is performed lazily (at call time) and cached so it only 
    runs once per session for the decorated object.

    Parameters
    ----------
    packages : dict, optional
        A dictionary mapping import names to minimum version strings.
        Example: ``{'awkward': '2.0.0', 'pyarrow': None}``.
        Use ``None`` if no specific version is required.

    Returns
    -------
    Callable
        The decorated function or class.

    Raises
    ------
    ImportError
        If a package is missing or the installed version is older than required.
    """
    if packages is None:
        packages = {}

    def decorator(func_or_class: Any) -> Any:
        # State to ensure we only check once per function/class
        # Using a list to allow mutation inside closures (compatible with older python)
        # or simply nonlocal in Python 3.
        state = {"checked": False}

        def check_deps():
            """Perform the dependency check if not already done."""
            if state["checked"]:
                return

            for pkg, min_ver in packages.items():
                # 1. Check if package exists
                spec = importlib.util.find_spec(pkg)
                if spec is None:
                    raise ImportError(
                        f"Missing optional dependency: '{pkg}'.\n"
                        f"This functionality requires {pkg}. Please install it via:\n\n"
                        f"    pip install {pkg}\n"
                    )
                
                # 2. Check version if specified
                if min_ver:
                    try:
                        module = __import__(pkg)
                        # Many packages store version in __version__
                        installed_ver = getattr(module, '__version__', None)
                        if installed_ver and version.parse(installed_ver) < version.parse(min_ver):
                            raise ImportError(
                                f"Package '{pkg}' is too old (found {installed_ver}, "
                                f"required >={min_ver}). Please upgrade."
                            )
                    except (AttributeError, ImportError):
                        # If we can't read the version, we assume it is okay 
                        pass
            
            # Mark as checked so we don't run this again
            state["checked"] = True

        if isinstance(func_or_class, type):
            # If decorating a class, wrap its __init__ method
            original_init = func_or_class.__init__
            
            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                check_deps()
                original_init(self, *args, **kwargs)
            
            func_or_class.__init__ = wrapped_init
            return func_or_class
        else:
            # If decorating a function
            @functools.wraps(func_or_class)
            def wrapper(*args, **kwargs):
                check_deps()
                return func_or_class(*args, **kwargs)
            return wrapper

    return decorator


class TableDisplay:
    """
    A lightweight helper to format data as tables.
    Respects global display configuration.
    """
    def __init__(self, headers, rows, title=None, total_rows=None):
        self.title = title
        self.total_rows = total_rows if total_rows is not None else len(rows)
        
        # Snapshot current config
        self.max_rows = global_display.max_rows
        self.max_colwidth = global_display.max_colwidth
        self.max_columns = global_display.max_columns
        self.display_width = global_display.display_width
        self.show_dimensions = global_display.show_dimensions
        self.float_precision = global_display.float_precision
        self.show_arrays = global_display.show_arrays
        self.max_array_elements = global_display.max_array_elements

        # Pre-process rows to filter arrays if needed
        self.headers, self.rows = self._filter_data(headers, rows)

    def _is_array(self, val):
        """Check if value is a list or numpy array."""
        return isinstance(val, (list, tuple, np.ndarray))

    def _filter_data(self, headers, rows):
        """Filter out array columns if show_arrays is False."""
        if not rows or self.show_arrays:
            return headers, rows

        # Detect array columns based on the first row
        valid_indices = []
        for i, val in enumerate(rows[0]):
            if not self._is_array(val):
                valid_indices.append(i)
        
        if len(valid_indices) == len(headers):
            return headers, rows
            
        new_headers = [headers[i] for i in valid_indices]
        new_rows = []
        for row in rows:
            new_rows.append([row[i] for i in valid_indices])
            
        return new_headers, new_rows

    def _format_value(self, val):
        """Helper to format a single value with precision and length control."""
        
        # 1. Handle Iterables (List, Tuple, Array)
        if isinstance(val, (list, tuple, np.ndarray)):
            # Unify access
            if isinstance(val, np.ndarray):
                # Flatten if multi-dimensional for simple display
                items = val.flatten() if val.ndim > 1 else val
            else:
                items = val
                
            n_items = len(items)
            limit = self.max_array_elements
            
            # Truncate logic
            truncated = False
            if n_items > limit:
                items_to_show = items[:limit]
                truncated = True
            else:
                items_to_show = items
            
            # Format individual items
            formatted_items = []
            for item in items_to_show:
                if isinstance(item, (float, np.floating)):
                    formatted_items.append(f"{item:.{self.float_precision}f}")
                else:
                    formatted_items.append(str(item))
            
            if truncated:
                formatted_items.append("...")
                
            content = ", ".join(formatted_items)
            
            if isinstance(val, tuple):
                return f"({content})"
            return f"[{content}]"

        # 2. Handle Scalar Floats
        if isinstance(val, (float, np.floating)):
            return f"{val:.{self.float_precision}f}"

        # 3. Fallback
        return str(val)

    def _format_ascii(self):
        """Generate an ASCII table with smart column truncation."""
        if not self.headers:
            return "(Empty or all columns hidden)"

        # 1. Determine Terminal Width
        target_width = self.display_width
        if target_width == 0:
            target_width = shutil.get_terminal_size((80, 20)).columns

        # 2. Prepare Sample Rows
        display_rows = self.rows
        is_row_truncated = len(self.rows) < self.total_rows
        row_split_idx = len(self.rows) // 2 if is_row_truncated else -1

        # 3. Calculate Widths
        n_cols = len(self.headers)
        col_widths = [len(h) for h in self.headers]
        
        formatted_data = []
        for row in display_rows:
            fmt_row = []
            for i, val in enumerate(row):
                s_val = self._format_value(val)
                # Truncate very long strings
                if len(s_val) > self.max_colwidth:
                    s_val = s_val[:self.max_colwidth-3] + "..."
                col_widths[i] = max(col_widths[i], len(s_val))
                fmt_row.append(s_val)
            formatted_data.append(fmt_row)

        # 4. Determine Visible Columns (Improved Logic)
        total_required_width = sum(col_widths) + (n_cols - 1) * 3
        visible_indices = []
        
        # If explicit limit set, enforce it
        if self.max_columns > 0 and n_cols > self.max_columns:
            # Force truncation based on count
            visible_indices = self._calculate_visible_indices(n_cols, self.max_columns)
            
        elif total_required_width <= target_width:
            # FITS: Show all columns
            visible_indices = list(range(n_cols))
            
        else:
            # DOES NOT FIT: Iteratively remove columns from the middle until it fits
            current_width = 0
            # Always keep ID (col 0)
            left_idx = 0
            # Always keep last column
            right_idx = n_cols - 1
            
            # Start with minimal set: [0, ..., N-1]
            # Base width = width(0) + width(...) + width(N-1) + separators
            ellipsis_width = 3
            current_width = col_widths[0] + 3 + ellipsis_width + 3 + col_widths[-1]
            
            if current_width > target_width:
                # Terminal is extremely narrow, just show what we can
                visible_indices = [0, -1, n_cols-1]
            else:
                # Try to add more columns from left and right
                # We use a deque-like approach: 0, 1, 2 ... N-3, N-2, N-1
                added_indices = {0, n_cols-1}
                candidate_left = 1
                candidate_right = n_cols - 2
                
                while candidate_left < candidate_right:
                    # Try adding left
                    w_add = col_widths[candidate_left] + 3 # + separator
                    if current_width + w_add <= target_width:
                        added_indices.add(candidate_left)
                        current_width += w_add
                        candidate_left += 1
                    else:
                        break
                        
                    # Try adding right
                    if candidate_left < candidate_right:
                        w_add = col_widths[candidate_right] + 3
                        if current_width + w_add <= target_width:
                            added_indices.add(candidate_right)
                            current_width += w_add
                            candidate_right -= 1
                        else:
                            break
                
                # Reconstruct ordered list with ellipsis
                sorted_ids = sorted(list(added_indices))
                visible_indices = []
                last_idx = -1
                for idx in sorted_ids:
                    if last_idx != -1 and idx > last_idx + 1:
                        visible_indices.append(-1)
                    visible_indices.append(idx)
                    last_idx = idx

        # 5. Format Output
        def get_w(idx):
            return col_widths[idx] if idx >= 0 else 3

        header_parts = []
        for idx in visible_indices:
            if idx == -1:
                header_parts.append("...")
            else:
                header_parts.append(f"{self.headers[idx]:<{col_widths[idx]}}")
        
        header_str = " | ".join(header_parts)
        separator = "-+-".join("-" * get_w(idx) for idx in visible_indices)

        lines = []
        if self.title:
            lines.append(f"[{self.title}]")
        lines.append(header_str)
        lines.append(separator)

        for i, row_data in enumerate(formatted_data):
            if is_row_truncated and i == row_split_idx:
                lines.append("..." .center(len(header_str)))
            
            row_parts = []
            for idx in visible_indices:
                if idx == -1:
                    row_parts.append("...")
                else:
                    row_parts.append(f"{row_data[idx]:<{col_widths[idx]}}")
            
            lines.append(" | ".join(row_parts))

        if self.show_dimensions and (is_row_truncated or (len(visible_indices) < n_cols and -1 in visible_indices)):
            lines.append(f"\n[{self.total_rows} rows x {n_cols} columns]")

        return "\n".join(lines)

    def _calculate_visible_indices(self, total, max_cols):
        """Helper for explicit column limit truncation."""
        if total <= max_cols:
            return list(range(total))
        
        # Keep first X, last 1
        # e.g. max=5 -> 0, 1, 2, ..., N-1
        keep_left = max(1, max_cols - 1)
        indices = list(range(keep_left))
        indices.append(-1)
        indices.append(total - 1)
        return indices

    def _format_html(self):
        """
        Generate an HTML table for Jupyter.
        Uses class="dataframe" to inherit native Jupyter/Pandas styling.
        """
        if not self.headers:
            return "<em>(Empty or all columns hidden)</em>"

        is_truncated = len(self.rows) < self.total_rows
        split_idx = len(self.rows) // 2 if is_truncated else -1

        # Container
        # Added font-family to ensure consistency if not inherited
        html = ['<div style="overflow: auto; font-family: sans-serif;">']
        
        if self.title:
            # UPDATED STYLE: Matches EventSummaryDisplay (uppercase, bold, opacity)
            html.append(
                f'<div style="margin-bottom: 8px; font-weight: bold; text-transform: uppercase; font-size: 0.85em; opacity: 0.8;">'
                f'{self.title}'
                f'</div>'
            )
        
        # Table
        html.append('<table border="1" class="dataframe" style="border: none; border-collapse: collapse; border-spacing: 0; font-size: 0.9em;">')
        
        # Header
        html.append('<thead><tr style="text-align: right;">')
        for h in self.headers:
            html.append(f'<th style="padding: 5px 10px;">{h}</th>')
        html.append('</tr></thead>')

        # Body
        html.append('<tbody>')
        for i, row in enumerate(self.rows):
            if is_truncated and i == split_idx:
                html.append(f'<tr><td colspan="{len(self.headers)}" style="text-align:center; font-style:italic;">...</td></tr>')
            
            html.append('<tr>')
            for val in row:
                s_val = self._format_value(val)
                html.append(f'<td style="padding: 5px 10px;">{s_val}</td>')
            html.append('</tr>')
        html.append('</tbody></table>')
        
        if is_truncated:
            html.append(f'<p style="font-size: 0.8em; color: var(--vscode-descriptionForeground, #666); margin-top: 5px;">{self.total_rows} rows x {len(self.headers)} columns</p>')
        html.append('</div>')
        
        return "".join(html)

    def __repr__(self):
        return self._format_ascii()

    def _repr_html_(self):
        return self._format_html()