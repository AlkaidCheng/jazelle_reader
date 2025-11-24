"""
General utility functions and decorators for the Jazelle package.
"""

import functools
import importlib.util
from packaging import version
from typing import Dict, Optional, Callable, Any

def requires_packages(packages: Optional[Dict[str, str]] = None) -> Callable:
    """
    Decorator to enforce optional dependencies for functions or classes.
    
    The check is performed lazily (at call time) and cached so it only 
    runs once per session for the decorated object.

    Parameters
    ----------
    packages : dict, optional
        A dictionary mapping import names to minimum version strings.
        Example: ``{'awkward': '2.0.0', 'pyarrow': None}``.
        Use ``None`` if no specific version is required.

    Returns
    -------
    Callable
        The decorated function or class.

    Raises
    ------
    ImportError
        If a package is missing or the installed version is older than required.
    """
    if packages is None:
        packages = {}

    def decorator(func_or_class: Any) -> Any:
        # State to ensure we only check once per function/class
        # Using a list to allow mutation inside closures (compatible with older python)
        # or simply nonlocal in Python 3.
        state = {"checked": False}

        def check_deps():
            """Perform the dependency check if not already done."""
            if state["checked"]:
                return

            for pkg, min_ver in packages.items():
                # 1. Check if package exists
                spec = importlib.util.find_spec(pkg)
                if spec is None:
                    raise ImportError(
                        f"Missing optional dependency: '{pkg}'.\n"
                        f"This functionality requires {pkg}. Please install it via:\n\n"
                        f"    pip install {pkg}\n"
                    )
                
                # 2. Check version if specified
                if min_ver:
                    try:
                        module = __import__(pkg)
                        # Many packages store version in __version__
                        installed_ver = getattr(module, '__version__', None)
                        if installed_ver and version.parse(installed_ver) < version.parse(min_ver):
                            raise ImportError(
                                f"Package '{pkg}' is too old (found {installed_ver}, "
                                f"required >={min_ver}). Please upgrade."
                            )
                    except (AttributeError, ImportError):
                        # If we can't read the version, we assume it is okay 
                        pass
            
            # Mark as checked so we don't run this again
            state["checked"] = True

        if isinstance(func_or_class, type):
            # If decorating a class, wrap its __init__ method
            original_init = func_or_class.__init__
            
            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                check_deps()
                original_init(self, *args, **kwargs)
            
            func_or_class.__init__ = wrapped_init
            return func_or_class
        else:
            # If decorating a function
            @functools.wraps(func_or_class)
            def wrapper(*args, **kwargs):
                check_deps()
                return func_or_class(*args, **kwargs)
            return wrapper

    return decorator