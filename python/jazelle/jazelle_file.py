"""
User-facing JazelleFile class.
"""

from typing import Optional, Dict, Any, List
from .jazelle_cython import JazelleFile as _JazelleFileCython
from .jazelle_cython import JazelleEvent
from .streamers.base import Streamer
from .converters import dict_to_awkward
from .utils import TableDisplay

# Import streamers to ensure registration and access explicit classes
from .streamers import ParquetStreamer, HDF5Streamer, JSONStreamer, FeatherStreamer

class JazelleFile(_JazelleFileCython):
    """
    Reader for SLD Jazelle files.

    This class extends the efficient C++ reader with native Awkward Array conversion,
    metadata management, and multi-format export capabilities.

    Examples
    --------
    >>> import jazelle
    >>> with jazelle.open("run123.jazelle") as f:
    ...     # Convert to Awkward Array
    ...     events = f.to_arrays()
    ...
    ...     # Save to Parquet with compression
    ...     f.to_parquet("output.parquet", compression="zstd")
    """

    def display(self, start: int = 0, count: int = 5, banks: Optional[List[str]] = None):
        """
        Display a summary table of events.
        
        Parameters
        ----------
        start : int
            Start index. Negative values count from the end.
        count : int
            Number of events to display.
        banks : list[str], optional
            If provided, adds columns showing the *count* of items in these 
            banks for each event (e.g. ['MCPART'] shows 'n_MCPART').
        """
        total = len(self)
        
        if start < 0:
            start += total
    
        if start < 0:
            start = 0
        elif start >= total:
            start = total

        remaining = total - start
        if count > remaining:
            count = remaining
        if count < 0:
            count = 0

        if count == 0:
            title = "Events (Empty)"
        else:
            end_idx = start + count - 1
            title = f"Events [{start + 1} - {end_idx + 1}]"

        return self._make_header_table(start, count, title, banks=banks)

    def head(self, n: int = 5, banks: Optional[List[str]] = None):
        """
        Display header information for the first n events.
        
        Parameters
        ----------
        n : int
            Number of events.
        banks : list[str], optional
            Additional bank families to show counts for.
        """
        return self.display(start=0, count=n, banks=banks)

    def tail(self, n: int = 5, banks: Optional[List[str]] = None):
        """
        Display header information for the last n events.
        """
        total = len(self)
        start = max(0, total - n)
        return self.display(start=start, count=n, banks=banks)

    def info(self, banks: Optional[List[str]] = None):
        """
        Print summary of file metadata and available bank types.
        
        Parameters
        ----------
        banks : list[str], optional
            If provided, only lists these banks if they are supported.
        """
        meta = self.metadata
        print("-" * 40)
        print(f"JazelleFile Info")
        print("-" * 40)
        print(f"File         : {meta.get('filename')}")
        print(f"Events       : {len(self)}")
        print(f"Created      : {meta.get('creation_date')}")
        print(f"Modified     : {meta.get('modified_date')}")
        
        print("-" * 40)

    def _make_header_table(self, start: int, count: int, title: str, banks: Optional[List[str]] = None):
        """Internal helper to build the display table."""
        events = self.read_batch(start=start, count=count)
        
        # Base headers from IEVENTH
        headers = ['id', 'run', 'event', 'evttime', 'weight']
        
        # Add dynamic headers for bank counts if requested
        bank_cols = []
        if banks:
            for b in banks:
                bank_cols.append(b.upper())
                headers.append(f"n_{b.upper()}")

        rows = []
        for evt in events:
            h = evt.ieventh.to_dict()
            row = []
            for col in headers:
                if col.startswith("n_"):
                    b_name = col[2:].upper()
                    try:
                        fam = evt.getFamily(b_name)
                        row.append(len(fam))
                    except Exception:
                        row.append(0) # Bank not present in this event
                else:
                    row.append(h.get(col))
            
            rows.append(row)
            
        return TableDisplay(headers, rows, title=title)

    def to_arrays(
        self, 
        start: int = 0, 
        count: int = -1, 
        batch_size: int = 1000, 
        num_threads: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Read events and return them as an Awkward Array.

        This is the primary method for loading Jazelle data into a modern,
        vectorized format suitable for analysis.

        Parameters
        ----------
        start : int, default 0
            Index of the first event to read.
        count : int, default -1
            Number of events to read. If -1, reads all remaining events.
        batch_size : int, default 1000
            Number of events per read batch (optimization parameter).
        num_threads : int, optional
            Number of threads for parallel reading. Defaults to auto-detection.
        metadata : dict, optional
            Dictionary of file-level metadata (e.g., filename, creation date) to 
            attach to the array's parameters. If None, uses ``self.metadata``.

        Returns
        -------
        awkward.Array
            A high-level array containing the event data.
        """
        if metadata is None:
            metadata = self.metadata

        data_dict = self.to_dict(
            layout='columnar',
            start=start,
            count=count,
            batch_size=batch_size,
            num_threads=num_threads
        )
        
        return dict_to_awkward(data_dict, metadata=metadata)

    # Alias for compatibility/preference
    to_awkward = to_arrays

    def save(
        self, 
        filename: str, 
        start: int = 0, 
        count: int = -1, 
        batch_size: int = 1000, 
        num_threads: Optional[int] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Export data to a file, inferring the format from the extension.

        Supported formats include: ``.parquet``, ``.h5``/``.hdf5``, ``.json``, ``.feather``.

        Parameters
        ----------
        filename : str
            Output filename (e.g. "data.parquet"). The extension determines the format.
        start : int, default 0
            Start event index.
        count : int, default -1
            Number of events to export.
        batch_size : int, default 1000
            Read batch size.
        num_threads : int, optional
            Parallel reading threads.
        metadata : dict, optional
            Metadata dictionary to attach/store. If None, uses ``self.metadata``.
        **kwargs
            Format-specific options passed to the writer (e.g., ``compression="lz4"``).
        """
        if metadata is None:
            metadata = self.metadata

        streamer = Streamer.get_streamer(filename)

        read_kwargs = dict(
            start=start,
            count=count,
            batch_size=batch_size,
            num_threads=num_threads
        )
        
        if isinstance(streamer, HDF5Streamer):
            data = self.to_dict(layout='columnar', **read_kwargs)
            streamer.dump(data, filename, metadata=metadata, **kwargs)
        else:
            dict_data = self.to_dict(layout='columnar', **read_kwargs)
            data = dict_to_awkward(dict_data, metadata=metadata)
            streamer.dump(data, filename, **kwargs)

    def to_parquet(
        self, 
        filename: str, 
        start: int = 0, 
        count: int = -1, 
        batch_size: int = 1000, 
        num_threads: Optional[int] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Export to Parquet format.

        Parameters
        ----------
        filename : str
            Output filename.
        start : int, default 0
            Start event index.
        count : int, default -1
            Number of events to export.
        batch_size : int, default 1000
            Read batch size.
        num_threads : int, optional
            Parallel reading threads.
        metadata : dict, optional
            Metadata dictionary to attach. If None, uses ``self.metadata``.
        **kwargs
            Arguments passed to ``awkward.to_parquet`` (e.g., ``compression``).
        """
        if metadata is None: metadata = self.metadata
        
        data = self.to_arrays(start, count, batch_size, num_threads, metadata=metadata)
        ParquetStreamer().dump(data, filename, **kwargs)

    def to_hdf5(
        self, 
        filename: str, 
        start: int = 0, 
        count: int = -1, 
        batch_size: int = 1000, 
        num_threads: Optional[int] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Export to HDF5 format.

        Parameters
        ----------
        filename : str
            Output filename.
        start : int, default 0
            Start event index.
        count : int, default -1
            Number of events to export.
        batch_size : int, default 1000
            Read batch size.
        num_threads : int, optional
            Parallel reading threads.
        metadata : dict, optional
            Metadata dictionary to store in file attributes. If None, uses ``self.metadata``.
        **kwargs
            Arguments passed to ``h5py.create_dataset``.
        """
        if metadata is None: metadata = self.metadata
        
        # HDF5 writer prefers the raw dictionary for layout control
        data = self.to_dict(
            layout='columnar', 
            start=start, 
            count=count, 
            batch_size=batch_size, 
            num_threads=num_threads
        )
        HDF5Streamer().dump(data, filename, metadata=metadata, **kwargs)

    def to_json(
        self, 
        filename: str, 
        start: int = 0, 
        count: int = -1, 
        batch_size: int = 1000, 
        num_threads: Optional[int] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Export to JSON format.

        Parameters
        ----------
        filename : str
            Output filename.
        start : int, default 0
            Start event index.
        count : int, default -1
            Number of events to export.
        batch_size : int, default 1000
            Read batch size.
        num_threads : int, optional
            Parallel reading threads.
        metadata : dict, optional
            Metadata dictionary. If None, uses ``self.metadata``.
        **kwargs
            Arguments passed to ``awkward.to_json``.
        """
        if metadata is None: metadata = self.metadata
        
        data = self.to_arrays(start, count, batch_size, num_threads, metadata=metadata)
        JSONStreamer().dump(data, filename, **kwargs)

    def to_feather(
        self, 
        filename: str, 
        start: int = 0, 
        count: int = -1, 
        batch_size: int = 1000, 
        num_threads: Optional[int] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Export to Feather format.

        Parameters
        ----------
        filename : str
            Output filename.
        start : int, default 0
            Start event index.
        count : int, default -1
            Number of events to export.
        batch_size : int, default 1000
            Read batch size.
        num_threads : int, optional
            Parallel reading threads.
        metadata : dict, optional
            Metadata dictionary. If None, uses ``self.metadata``.
        **kwargs
            Arguments passed to ``awkward.to_feather``.
        """
        if metadata is None: metadata = self.metadata
        
        data = self.to_arrays(start, count, batch_size, num_threads, metadata=metadata)
        FeatherStreamer().dump(data, filename, **kwargs)