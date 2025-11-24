import abc
import os
from typing import List, Any, Optional, Dict
from ..utils import requires_packages

# Global Registry: extension (str) -> Streamer Class
_EXTENSION_REGISTRY = {}

class Streamer(abc.ABC):
    """
    Abstract base class for Data Streamers.
    """
    
    # List of file extensions this streamer handles (e.g., ['.json'])
    extensions: List[str] = []

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses based on their extensions."""
        super().__init_subclass__(**kwargs)
        for ext in cls.extensions:
            # Normalize extension to .lower
            clean_ext = ext.lower() if ext.startswith('.') else '.' + ext.lower()
            _EXTENSION_REGISTRY[clean_ext] = cls

    @classmethod
    def get_streamer(cls, filename: str) -> 'Streamer':
        """
        Factory method to get the appropriate streamer for a filename.

        Parameters
        ----------
        filename : str
            The filename to check extension against.

        Returns
        -------
        Streamer
            An instance of the matching streamer class.

        Raises
        ------
        ValueError
            If no streamer is registered for the file extension.
        """
        _, ext = os.path.splitext(filename)
        streamer_cls = _EXTENSION_REGISTRY.get(ext.lower())
        
        if streamer_cls is None:
            raise ValueError(f"No streamer registered for file extension '{ext}'")
        
        return streamer_cls()

    @abc.abstractmethod
    def read(self, filename: str, **kwargs) -> Any:
        """
        Read file and return data.

        Parameters
        ----------
        filename : str
            Path to the file.
        **kwargs
            Arguments passed to the underlying reader.

        Returns
        -------
        Any
            Typically an ``awkward.Array``.
        """
        pass

    @abc.abstractmethod
    def dump(self, data: Any, filename: str, metadata: Optional[Dict] = None, **kwargs) -> None:
        """
        Write data to file.

        Parameters
        ----------
        data : Any
            The data to write (dict or awkward.Array).
        filename : str
            Output path.
        metadata : dict, optional
            Dictionary of file-level metadata to embed in the output.
        **kwargs
            Arguments passed to the underlying writer.
        """
        pass

    def _ensure_awkward(self, data: Any, metadata: Optional[Dict] = None):
        """
        Helper to ensure data is an Awkward Array before dumping.
        
        If data is a dictionary (from C++), it converts it.
        If data is already an Array, it attaches the metadata parameters.
        """
        import awkward as ak
        
        if isinstance(data, dict):
            # Import locally to avoid circular dependency at module level
            from ..converters import dict_to_awkward
            return dict_to_awkward(data, metadata=metadata)
            
        elif isinstance(data, ak.Array):
            if metadata:
                # Attach metadata parameters to existing array
                # We loop because with_parameter creates a new wrapper each time
                current_arr = data
                for k, v in metadata.items():
                    current_arr = ak.with_parameter(current_arr, k, v)
                return current_arr
            return data
            
        else:
            raise TypeError(f"Data must be a dict (from to_dict) or awkward.Array, not {type(data)}")