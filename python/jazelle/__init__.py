"""
Jazelle - A fast reader for SLD Jazelle files
"""

from ._version import __version__

# Core Classes
from .jazelle_cython import JazelleEvent, Family
from .jazelle_file import JazelleFile

# IO Functions
from .streamers.parquet_streamer import to_parquet, from_parquet
from .streamers.hdf5_streamer import to_hdf5, from_hdf5
from .streamers.json_streamer import to_json, from_json
from .streamers.feather_streamer import to_feather, from_feather

# Data Structures
from .jazelle_cython import (
    PIDVEC, CRIDHYP,
    IEVENTH, MCHEAD, MCPART, PHPSUM, PHCHRG, 
    PHKLUS, PHWIC, PHCRID, PHKTRK, PHKELID
)

def open(filename: str, **kwargs) -> JazelleFile:
    """
    Open a Jazelle file for reading.
    
    Parameters
    ----------
    filename : str
        Path to the file.
    **kwargs
        Arguments passed to the ``JazelleFile`` constructor 
        (e.g., ``num_threads``).
    
    Returns
    -------
    JazelleFile
        The open file object.
    """
    return JazelleFile(filename, **kwargs)

__all__ = [
    '__version__',
    'open',
    'JazelleFile',
    'JazelleEvent',
    'Family',
    
    # IO
    'to_parquet', 'from_parquet',
    'to_hdf5', 'from_hdf5',
    'to_json', 'from_json',
    'to_feather', 'from_feather',
    
    # Banks
    'IEVENTH', 'MCHEAD', 'MCPART', 'PHPSUM', 'PHCHRG', 
    'PHKLUS', 'PHWIC', 'PHCRID', 'PHKTRK', 'PHKELID',
    'PIDVEC', 'CRIDHYP'
]