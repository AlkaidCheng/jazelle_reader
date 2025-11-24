"""
Jazelle - A fast reader for SLD Jazelle files
"""

from ._version import __version__

from .jazelle_cython import JazelleFile, JazelleEvent

from .jazelle_cython import (
    # Helper structs
    PIDVEC,
    CRIDHYP,
    
    # Bank types
    IEVENTH,
    MCHEAD,
    MCPART,
    PHPSUM,
    PHCHRG,
    PHKLUS,
    PHWIC,
    PHCRID,
    PHKTRK,
    PHKELID,
    
    # Family container
    Family,
)

__all__ = [
    # Version
    '__version__',
    
    # Main classes
    'JazelleFile',
    'JazelleEvent',
    'Family',
    
    # Helper structs
    'PIDVEC',
    'CRIDHYP',
    
    # Bank types
    'IEVENTH',
    'MCHEAD',
    'MCPART',
    'PHPSUM',
    'PHCHRG',
    'PHKLUS',
    'PHWIC',
    'PHCRID',
    'PHKTRK',
    'PHKELID',
]