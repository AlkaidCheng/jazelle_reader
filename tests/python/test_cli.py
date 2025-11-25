import subprocess
import sys
import pytest

def test_cli_inspect(jazelle_data_path):
    """Test 'jazelle inspect' command."""
    cmd = [sys.executable, "-m", "jazelle.cli", "inspect", jazelle_data_path, "-n", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "JazelleFile Info" in result.stdout
    assert "Events" in result.stdout

def test_cli_convert_parquet(jazelle_data_path, tmp_path):
    """Test 'jazelle convert' command."""
    output = tmp_path / "cli_test.parquet"
    cmd = [
        sys.executable, "-m", "jazelle.cli", "convert",
        "-i", jazelle_data_path,
        "-o", str(output),
        "--count", "5"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert output.exists()