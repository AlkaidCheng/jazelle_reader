from typing import Any, Dict, Optional
from .base import Streamer
from ..utils import requires_packages

@requires_packages({'awkward': '2.0.0', 'pyarrow': None})
class ParquetStreamer(Streamer):
    """
    Streamer for Apache Parquet format.
    """
    extensions = ['.parquet']

    def read(self, filename: str, **kwargs) -> Any:
        import awkward as ak
        return ak.from_parquet(filename, **kwargs)

    def dump(
        self, 
        data: Any, 
        filename: str, 
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> None:
        import awkward as ak
        ak_array = self._ensure_awkward(data, metadata=metadata)
        ak.to_parquet(ak_array, filename, **kwargs)

def from_parquet(filename: str, **kwargs):
    """
    Read a Parquet file into an Awkward Array.

    Parameters
    ----------
    filename : str
        Path to the input file.
    **kwargs
        Arguments passed to ``awkward.from_parquet``.

    Returns
    -------
    awkward.Array
    """
    return ParquetStreamer().read(filename, **kwargs)

def to_parquet(data: Any, filename: str, metadata: Optional[Dict] = None, **kwargs):
    """
    Write data to a Parquet file.

    Parameters
    ----------
    data : dict or awkward.Array
        Data to serialize.
    filename : str
        Output path.
    metadata : dict, optional
        Metadata to attach to the array.
    **kwargs
        Arguments passed to ``awkward.to_parquet``.
    """
    return ParquetStreamer().dump(data, filename, metadata=metadata, **kwargs)