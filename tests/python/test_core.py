import pytest
import numpy as np
import jazelle

def test_file_metadata(jazelle_data_path):
    with jazelle.open(jazelle_data_path) as f:
        assert f.fileName is not None
        assert f.getTotalEvents() > 0

def test_read_batch_vs_sequential(jazelle_data_path):
    """Ensure parallel reading yields same result as sequential."""
    with jazelle.open(jazelle_data_path) as f:
        # Read first 10 events sequentially
        seq_events = [f[i] for i in range(10)]
        
        # Read batch parallel
        batch_events = f.read_batch(0, 10, num_threads=2)
        
        assert len(seq_events) == len(batch_events)
        
        # Compare specific values (e.g., run number)
        for s, b in zip(seq_events, batch_events):
            assert s.ieventh.run == b.ieventh.run
            assert s.ieventh.event == b.ieventh.event

def test_to_dict_structure(jazelle_data_path):
    with jazelle.open(jazelle_data_path) as f:
        data = f.to_dict(start=0, count=5, layout='columnar')
        
        assert 'IEVENTH' in data
        assert 'run' in data['IEVENTH']
        assert len(data['IEVENTH']['run']) == 5
        
        # Check offsets exist for families
        if 'MCPART' in data:
            assert '_offsets' in data['MCPART']