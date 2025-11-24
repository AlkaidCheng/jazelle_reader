"""
Utilities for converting internal data structures to third-party formats.
"""
from typing import Dict, Tuple, Any, Optional
from .utils import requires_packages

@requires_packages({'awkward': '2.0.0'})
def dict_to_awkward(data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
    """
    Convert a Jazelle dictionary (columnar layout) to an Awkward Array.

    This reconstructs the nested structure from flat arrays and offset arrays
    produced by the C++ reader.

    Parameters
    ----------
    data : dict
        The dictionary output from ``JazelleFile.to_dict(layout='columnar')``.
    metadata : dict, optional
        Dictionary of file-level metadata (e.g., filename, creation date) to 
        attach to the array's parameters.

    Returns
    -------
    awkward.Array
        A high-level array representing the event data.
    """
    import awkward as ak
    
    event_fields = []
    event_contents = []
    
    for family_name, family_data in data.items():
        # Skip non-family entries (like scalar metadata if any)
        if not isinstance(family_data, dict) or not family_data:
            continue
            
        # 1. Separate offsets from data columns
        offsets = family_data.get('_offsets')
        columns = {k: v for k, v in family_data.items() if k != '_offsets'}
        
        if not columns:
            continue

        # 2. Create the Flat Content (RecordArray)
        # Wrap numpy arrays in ak.contents.NumpyArray (Zero-Copy where possible)
        col_names = list(columns.keys())
        col_contents = [ak.contents.NumpyArray(columns[name]) for name in col_names]
        
        # Determine length from the first column
        content_len = len(col_contents[0])
        
        flat_record = ak.contents.RecordArray(
            col_contents, 
            col_names, 
            length=content_len
        )

        # 3. Apply Structure (ListOffsetArray vs Regular Array)
        if offsets is not None:
            # Jagged data: Use offsets to define boundaries
            ak_offsets = ak.index.Index64(offsets)
            content = ak.contents.ListOffsetArray(ak_offsets, flat_record)
        else:
            # Singleton data (e.g. IEVENTH): One entry per event
            content = flat_record
            
        event_fields.append(family_name)
        event_contents.append(content)

    # 4. Create Final Array
    if not event_contents:
        return ak.Array([])

    # All top-level families must have the same length (number of events)
    n_events = len(event_contents[0])
    
    # Create final Event record with metadata parameters
    layout = ak.contents.RecordArray(
        event_contents, 
        event_fields, 
        length=n_events,
        parameters=metadata
    )
    
    return ak.Array(layout)

@requires_packages({'awkward': '2.0.0'})
def awkward_to_dict(array) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Convert an Awkward Array back to a Jazelle dictionary (columnar layout).

    This allows Awkward Arrays to be converted back into the format required
    by the HDF5Streamer or other internal tools.

    Parameters
    ----------
    array : awkward.Array
        The input array to convert.

    Returns
    -------
    tuple
        (data_dict, metadata)
    """
    import awkward as ak
    import numpy as np
    
    if not isinstance(array, ak.Array):
        raise TypeError(f"Input must be an awkward.Array, not {type(array)}")

    data = {}
    metadata = array.layout.parameters or None
    
    for family_name in array.fields:
        family_content = array[family_name]
        family_dict = {}
        
        layout = family_content.layout
        
        # 1. Handle Jagged Arrays (ListOffsetArray)
        if isinstance(layout, ak.contents.ListOffsetArray):
            # Extract offsets (Zero-copy view if possible)
            family_dict['_offsets'] = np.asarray(layout.offsets)
            
            # Drill down to the flat content
            record_content = layout.content
        else:
            # Handle Singleton/Flat Arrays
            record_content = layout
            
        # 2. Extract Columns from RecordArray
        if isinstance(record_content, ak.contents.RecordArray):
            for i, field in enumerate(record_content.fields):
                content = record_content.content(i)
                
                # We expect flat NumpyArrays here
                if isinstance(content, ak.contents.NumpyArray):
                    family_dict[field] = np.asarray(content.data)
                else:
                    # Fallback: try to convert nested/complex columns to numpy
                    try:
                        family_dict[field] = ak.to_numpy(ak.Array(content))
                    except Exception:
                        pass
        
        if family_dict:
            data[family_name] = family_dict
            
    return data, metadata