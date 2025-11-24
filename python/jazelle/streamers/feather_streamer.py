from typing import Any, Union, Dict, Optional
from .base import Streamer
from ..utils import requires_packages

@requires_packages({'awkward': '2.0.0', 'pyarrow': None})
class FeatherStreamer(Streamer):
    """
    Streamer for Apache Arrow Feather format.
    """
    extensions = ['.feather', '.arrow']

    def read(self, filename: str, **kwargs) -> Any:
        """
        Read a Feather/Arrow file into an Awkward Array.

        Parameters
        ----------
        filename : str
            Path to the input file.
        **kwargs
            Additional arguments passed to ``awkward.from_feather``.

        Returns
        -------
        awkward.Array
            The loaded data.
        """
        import awkward as ak
        return ak.from_feather(filename, **kwargs)

    def dump(self, data: Any, filename: str, metadata: Optional[Dict] = None, **kwargs) -> None:
        """
        Write data to a Feather/Arrow file.

        Parameters
        ----------
        data : Any
            The data to write (dict from ``to_dict`` or ``awkward.Array``).
        filename : str
            Path to the output file.
		metadata : dict, optional
			Metadata to attach to the array.
        **kwargs
            Additional arguments passed to ``awkward.to_feather`` 
            (e.g., ``compression="zstd"``).
        """
        import awkward as ak
        ak_array = self._ensure_awkward(data, metadata)
        ak.to_feather(ak_array, filename, **kwargs)

def from_feather(filename: str, **kwargs):
    """
    Read a Feather file into an Awkward Array.

    Parameters
    ----------
    filename : str
        Path to the input file.
    **kwargs
        Arguments passed to ``awkward.from_feather``.

    Returns
    -------
    awkward.Array
    """
    return FeatherStreamer().read(filename, **kwargs)

def to_feather(data: Any, filename: str, metadata: Optional[Dict] = None, **kwargs):
    """
    Write data to a Feather file.

    Parameters
    ----------
    data : dict or awkward.Array
        Data to serialize.
    filename : str
        Output path.
    metadata : dict, optional
        Metadata to attach to the array.
    **kwargs
        Arguments passed to ``awkward.to_feather``.
    """
    return FeatherStreamer().dump(data, filename, metadata=metadata, **kwargs)