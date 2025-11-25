import pytest
import os
import jazelle
import awkward as ak

@pytest.mark.parametrize("fmt", ["parquet", "h5", "json"])
def test_export_cycle(jazelle_data_path, tmp_path, fmt):
    """Test Read -> Write -> Read Back cycle."""
    output_file = tmp_path / f"output.{fmt}"
    
    # 1. Write
    with jazelle.open(jazelle_data_path) as f:
        # Process small chunk
        if fmt == "h5":
            f.to_hdf5(str(output_file), start=0, count=10)
        elif fmt == "parquet":
            f.to_parquet(str(output_file), start=0, count=10)
        elif fmt == "json":
            f.to_json(str(output_file), start=0, count=10)

    assert output_file.exists()
    assert output_file.stat().st_size > 0

    # 2. Read Back (using specific streamers)
    if fmt == "parquet":
        arr = jazelle.from_parquet(str(output_file))
        assert len(arr) == 10
        assert "IEVENTH" in arr.fields
    elif fmt == "h5":
        arr = jazelle.from_hdf5(str(output_file))
        assert len(arr) == 10