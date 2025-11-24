from typing import Any, Dict, Optional
from .base import Streamer
from ..utils import requires_packages

@requires_packages({'awkward': '2.0.0'})
class JSONStreamer(Streamer):
    """
    Streamer for JSON format.
    """
    extensions = ['.json']

    def read(self, filename: str, **kwargs) -> Any:
        """
        Read a JSON file into an Awkward Array.

        Parameters
        ----------
        filename : str
            Path to the input JSON file.
        **kwargs
            Additional arguments passed to ``awkward.from_json``.

        Returns
        -------
        awkward.Array
            The loaded data.
        """
        import awkward as ak
        return ak.from_json(filename, **kwargs)

    def dump(self, data: Any, filename: str, metadata: Optional[Dict] = None, **kwargs) -> None:
        """
        Write data to a JSON file.

        Parameters
        ----------
        data : Any
            The data to write (dict from ``to_dict`` or ``awkward.Array``).
        filename : str
            Path to the output JSON file.
        metadata : dict, optional
            Metadata to attach to the array.
        **kwargs
            Additional arguments passed to ``awkward.to_json`` (e.g., ``line_delimited=True``).
        """
        import awkward as ak
        ak_array = self._ensure_awkward(data, metadata=metadata)
        ak.to_json(ak_array, filename, **kwargs)

def from_json(filename: str, **kwargs):
    """
    Read a JSON file into an Awkward Array.

    Parameters
    ----------
    filename : str
        Path to the input file.
    **kwargs
        Arguments passed to ``awkward.from_json``.

    Returns
    -------
    awkward.Array
    """
    return JSONStreamer().read(filename, **kwargs)

def to_json(data: Any, filename: str, metadata: Optional[Dict] = None, **kwargs):
    """
    Write data to a JSON file.

    Parameters
    ----------
    data : dict or awkward.Array
        Data to serialize.
    filename : str
        Output path.
    metadata : dict, optional
        Metadata to attach to the array.
    **kwargs
        Arguments passed to ``awkward.to_json``.
    """
    return JSONStreamer().dump(data, filename, metadata=metadata, **kwargs)