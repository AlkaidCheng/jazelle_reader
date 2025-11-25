import pytest
import re
from jazelle.utils import TableDisplay
import jazelle

# --- 1. Unit Tests for TableDisplay Utility (No Data File Needed) ---

def test_table_display_ascii():
    """Test basic ASCII table generation."""
    headers = ["ID", "Value", "Array"]
    rows = [
        [1, 10.5, [1, 2, 3]],
        [2, 20.0, [4, 5]]
    ]
    
    table = TableDisplay(headers, rows, title="Test Table")
    output = str(table)
    
    assert "[Test Table]" in output
    assert "ID" in output
    assert "10.50000" in output  # Default precision is 5
    assert "[1, 2, 3]" in output

def test_table_display_html():
    """Test HTML generation for Jupyter."""
    headers = ["ColA", "ColB"]
    rows = [[1, "text"]]
    
    table = TableDisplay(headers, rows)
    html = table._repr_html_()
    
    assert "<table" in html
    assert "ColA" in html
    assert "text" in html
    assert "class=\"dataframe\"" in html

def test_display_config_truncation():
    """Test that global config options actually affect output."""
    # Set restrictive options
    jazelle.set_display_options(max_rows=2, max_colwidth=10)
    
    headers = ["LongColumn"]
    rows = [["A very long string that should be truncated"]] * 5
    
    table = TableDisplay(headers, rows, total_rows=5)
    output = str(table)
    
    # Check row truncation (should see ellipsis for rows)
    assert output.count("\n") < 15  # Rough check for compactness
    # Check column truncation
    assert "..." in output 
    assert "A very..." in output or "should be..." not in output

    # Reset options
    jazelle.set_display_options(max_rows=10, max_colwidth=50)

# --- 2. Integration Tests with Jazelle Objects ---

def test_file_inspect_methods(jazelle_data_path, capsys):
    """Test file-level head/tail/info methods."""
    with jazelle.open(jazelle_data_path) as f:
        # Test info() (prints to stdout)
        f.info()
        captured = capsys.readouterr()
        assert "JazelleFile Info" in captured.out
        assert "Events" in captured.out

        # Test head()
        tbl_head = f.head(n=2)
        assert isinstance(tbl_head, TableDisplay)
        assert "Events [1 - 2]" in str(tbl_head)

        # Test tail()
        tbl_tail = f.tail(n=2)
        assert isinstance(tbl_tail, TableDisplay)
        total = len(f)
        assert f"Events [{total-1} - {total}]" in str(tbl_tail)

def test_event_display(sample_event):
    """Test single event summary display."""
    # ASCII
    output = str(sample_event)
    assert "Run" in output
    assert "Event" in output
    assert "Data Banks" in output
    
    # HTML
    html = sample_event._repr_html_()
    assert "display: flex" in html # The side-by-side layout style
    assert "Event Header" in html

def test_family_display(sample_event):
    """Test bank family table display."""
    # Find a non-empty family
    fam = None
    for f in sample_event.getFamilies():
        if len(f) > 0:
            fam = f
            break
    
    if fam:
        # ASCII
        output = str(fam)
        assert fam.name in output
        assert "id" in output
        
        # Check array formatting in table
        # e.g., if it's MCPART, it has 'p' array
        if hasattr(fam[0], 'p'):
            assert "[" in output or "(" in output

def test_display_with_bank_filtering(jazelle_data_path):
    """Test the 'banks' argument in file.display()."""
    with jazelle.open(jazelle_data_path) as f:
        # Request specific bank counts
        tbl = f.display(start=0, count=5, banks=['MCPART', 'PHCHRG'])
        output = str(tbl)
        
        assert "n_MCPART" in output or "N_MCPART" in output
        assert "n_PHCHRG" in output or "N_PHCHRG" in output