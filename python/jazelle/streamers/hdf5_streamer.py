from typing import Any, Dict, Optional
from .base import Streamer
from ..utils import requires_packages

@requires_packages({'h5py': None, 'numpy': None})
class HDF5Streamer(Streamer):
    """
    Streamer for HDF5 format.

    This streamer is optimized to write the internal C++ columnar layout directly 
    to HDF5 groups and datasets, providing high performance and compression.
    """
    extensions = ['.h5', '.hdf5']

    def read(self, filename: str, **kwargs) -> Any:
        """
        Read an HDF5 file (written by this streamer) into an Awkward Array.

        Parameters
        ----------
        filename : str
            Path to the input HDF5 file.
        **kwargs
            Arguments passed to ``h5py.File``.

        Returns
        -------
        awkward.Array
            The reconstructed event data with metadata attached.
        """
        import h5py
        from ..converters import dict_to_awkward

        data_dict = {}
        metadata = {}

        with h5py.File(filename, "r", **kwargs) as f:
            for k, v in f.attrs.items():
                metadata[k] = v

            # We look for the 'jazelle_events' group or use root if it contains families directly
            root = f
            if "jazelle_events" in f:
                root = f["jazelle_events"]

            for key in root.keys():
                obj = root[key]
                
                # We expect Groups representing Families (e.g., "MCPART")
                if isinstance(obj, h5py.Group):
                    family_name = key
                    family_data = {}

                    for col_name in obj.keys():
                        ds = obj[col_name]
                        if isinstance(ds, h5py.Dataset):
                            # Load into numpy array
                            family_data[col_name] = ds[:]
                    
                    if family_data:
                        data_dict[family_name] = family_data

        return dict_to_awkward(data_dict, metadata=metadata)

    def dump(
        self, 
        data: Dict[str, Any], 
        filename: str, 
        metadata: Optional[Dict[str, Any]] = None,
        group_name: str = "jazelle_events", 
        compression: str = "gzip", 
        **kwargs
    ) -> None:
        """
        Write data to an HDF5 file.

        Parameters
        ----------
        data : dict
            The data to write (must be the dictionary output from ``to_dict``).
        filename : str
            Path to the output HDF5 file.
        metadata : dict, optional
            Dictionary of file-level metadata to store in root attributes.
        group_name : str, default "jazelle_events"
            The root group name to store the event data under.
        compression : str, default "gzip"
            Compression filter to use for datasets (e.g., "gzip", "lzf").
        **kwargs
            Additional arguments passed to ``h5py.Group.create_dataset``.

        Raises
        ------
        TypeError
            If ``data`` is not a dictionary.
        """
        import h5py
        import awkward as ak
        
        # Convert Awkward Array to Dict if necessary
        if isinstance(data, ak.Array):
            from ..converters import awkward_to_dict
            data_dict, meta_from_arr = awkward_to_dict(data)
            # Merge metadata, preferring explicit argument
            if metadata is None:
                metadata = meta_from_arr
            data = data_dict
        
        elif not isinstance(data, dict):
            raise TypeError(
                 "HDF5Streamer requires a dictionary or awkward.Array."
             )

        with h5py.File(filename, "w") as f:
            
            # Write Metadata to Root Attributes
            if metadata:
                for k, v in metadata.items():
                    try:
                        f.attrs[k] = v
                    except TypeError:
                        # Fallback for complex objects -> string
                        f.attrs[k] = str(v)

            # Create the base group
            base_grp = f.create_group(group_name)
            
            for family_name, content in data.items():
                if not isinstance(content, dict): 
                    continue
                
                fam_grp = base_grp.create_group(family_name)
                
                for col_name, array in content.items():
                    fam_grp.create_dataset(
                        col_name, 
                        data=array, 
                        compression=compression, 
                        **kwargs
                    )

def from_hdf5(filename: str, **kwargs):
    """Read an HDF5 file (Not Implemented)."""
    return HDF5Streamer().read(filename, **kwargs)

def to_hdf5(data: Dict, filename: str, metadata: Optional[Dict] = None, **kwargs):
    """
    Write data to an HDF5 file.

    Parameters
    ----------
    data : dict
        Data to serialize (must be from ``to_dict``).
    filename : str
        Output path.
    metadata : dict, optional
        Metadata to store in attributes.
    **kwargs
        Arguments passed to ``h5py``.
    """
    return HDF5Streamer().dump(data, filename, metadata=metadata, **kwargs)