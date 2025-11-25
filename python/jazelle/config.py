"""
Global configuration for Jazelle display and behavior.
"""

class DisplayOptions:
    def __init__(self):
        # Maximum number of rows to display in tables (head + tail)
        self.max_rows = 10
        
        # Maximum width of columns in ASCII tables
        self.max_colwidth = 50
        
        # Maximum columns to display in ASCII tables (0 = auto/unlimited)
        self.max_columns = 0
        
        # Target width for ASCII display (0 = auto-detect terminal width)
        self.display_width = 0
        
        # Whether to show the "dimensions" footer (e.g. "100 rows x 5 cols")
        self.show_dimensions = True
        
        # Float precision for display (number of decimal places)
        self.float_precision = 5
        
        # Whether to show array/vector attributes (e.g. hlxpar[6]) in tables
        self.show_arrays = True
        
        # Maximum number of elements to show in an array before truncation
        self.max_array_elements = 3

# Singleton instance
display = DisplayOptions()

def set_display_options(max_rows=None, max_colwidth=None, max_columns=None, display_width=None, show_dimensions=None, float_precision=None, show_arrays=None, max_array_elements=None):
    """
    Set global display options.
    
    Parameters
    ----------
    max_rows : int, optional
        Max rows to display in family/event tables.
    max_colwidth : int, optional
        Max characters per column in ASCII output.
    max_columns : int, optional
        Max columns to display (0 for unlimited/auto).
    display_width : int, optional
        Target console width (0 for auto-detect).
    show_dimensions : bool, optional
        Whether to print table dimensions footer.
    float_precision : int, optional
        Number of decimal places for floating point numbers.
    show_arrays : bool, optional
        Whether to display columns containing array data (default: True).
    max_array_elements : int, optional
        Maximum elements to show in array strings (default: 5).
    """
    if max_rows is not None:
        display.max_rows = max_rows
    if max_colwidth is not None:
        display.max_colwidth = max_colwidth
    if max_columns is not None:
        display.max_columns = max_columns
    if display_width is not None:
        display.display_width = display_width
    if show_dimensions is not None:
        display.show_dimensions = show_dimensions
    if float_precision is not None:
        display.float_precision = float_precision
    if show_arrays is not None:
        display.show_arrays = show_arrays
    if max_array_elements is not None:
        display.max_array_elements = max_array_elements