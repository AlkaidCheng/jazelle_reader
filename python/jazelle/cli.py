import argparse
import sys
import os
import fnmatch
from typing import List, Optional

import jazelle

def parse_bank_patterns(patterns: Optional[List[str]]) -> Optional[List[str]]:
    """
    Expands wildcard patterns (e.g., 'MC*', '*') into a list of concrete 
    bank names supported by the library.
    
    The returned list respects the canonical order of banks defined in 
    JazelleEvent.getKnownBankNames().
    """
    if not patterns:
        return None

    known_banks = jazelle.JazelleEvent.getKnownBankNames()
    selected_banks_set = set()

    for pattern in patterns:
        # Case-insensitive matching
        matches = fnmatch.filter(known_banks, pattern.upper())
        if not matches:
            if pattern in known_banks:
                matches = [pattern]
            else:
                print(f"Warning: No banks matched pattern '{pattern}'", file=sys.stderr)
        selected_banks_set.update(matches)
    
    return [b for b in known_banks if b in selected_banks_set]

def parse_and_apply_display_options(options_str: Optional[str]):
    """
    Parses a display option string (e.g., "max_rows=5, show_arrays=0")
    and applies them to the global configuration.
    """
    if not options_str:
        return

    kwargs = {}
    try:
        settings = options_str.split(',')
        for setting in settings:
            setting = setting.strip()
            if not setting:
                continue
                
            if '=' not in setting:
                print(f"Warning: Invalid display option format '{setting}'. Expected 'key=value'.", file=sys.stderr)
                continue
                
            key, val_str = setting.split('=', 1)
            key = key.strip()
            val_str = val_str.strip()
            
            try:
                val = int(val_str)
            except ValueError:
                print(f"Warning: Invalid value for '{key}': '{val_str}'. Must be an integer.", file=sys.stderr)
                continue
                
            kwargs[key] = val

        if kwargs:
            jazelle.set_display_options(**kwargs)
            
    except Exception as e:
        print(f"Warning: Error parsing display options: {e}", file=sys.stderr)

def handle_inspect(args):
    """Handler for the 'inspect' command."""
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        parse_and_apply_display_options(args.display_options)
        resolved_banks = parse_bank_patterns(args.banks)
        num_threads = args.threads if args.threads is not None else 0
        
        with jazelle.open(args.input, num_threads=num_threads) as f:
            f.info(banks=resolved_banks)
            print("") # Spacer

            total = len(f)
            start = 0
            count = args.lines

            if args.tail:
                start = max(0, total - count)
                title_pfx = "Tail"
            else:
                title_pfx = "Head"

            if count > 0:
                table = f.display(start=start, count=count, banks=resolved_banks)
                print(table)

    except Exception as e:
        print(f"Error inspecting file: {e}", file=sys.stderr)
        sys.exit(1)

def handle_read(args):
    """Handler for the 'read' command."""
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        # Apply display options first
        parse_and_apply_display_options(args.display_options)

        with jazelle.open(args.input, num_threads=0) as f: # Sequential read usually fine here
            idx = args.index
            if idx < 0: 
                idx += len(f)
            
            if idx < 0 or idx >= len(f):
                print(f"Error: Event index {idx} out of bounds (0-{len(f)-1})", file=sys.stderr)
                sys.exit(1)

            # Read specific event
            event = f[idx]

            # Resolve wildcards
            resolved_banks = parse_bank_patterns(args.banks)

            output_parts = []

            if not resolved_banks:
                # Default: Show Summary (Header + Bank Counts)
                output_parts.append(str(event.display()))
            else:
                header = event.ieventh
                output_parts.append(f"Run: {header.run} | Event: {header.event} | Time: {header.evttime}")
                output_parts.append("-" * 60)

                for bank_name in resolved_banks:
                    try:
                        fam = event.getFamily(bank_name)
                        output_parts.append(f"\n[{bank_name} ({len(fam)} entries)]")
                        output_parts.append(str(fam))
                    except Exception as e:
                        print(f"Warning: Could not retrieve bank '{bank_name}': {e}", file=sys.stderr)

            output = "\n".join(output_parts)
            
            # Handle line limiting
            lines = output.splitlines()
            if args.limit > 0 and len(lines) > args.limit:
                print("\n".join(lines[:args.limit]))
                print(f"\n... [Output truncated after {args.limit} lines]")
            else:
                print(output)

    except Exception as e:
        print(f"Error reading event: {e}", file=sys.stderr)
        sys.exit(1)

def handle_convert(args):
    """Handler for the 'convert' command."""
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        
        with jazelle.open(args.input) as f:
            f.save(
                args.output,
                start=args.start,
                count=args.count,
                batch_size=args.batch_size,
                num_threads=args.threads
            )

    except Exception as e:
        print(f"Error converting file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    # Main parser configuration
    parser = argparse.ArgumentParser(
        description="Jazelle Data Reader CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # --- INSPECT Command ---
    p_inspect = subparsers.add_parser(
        'inspect', 
        help="Display file metadata and event summary table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p_inspect.add_argument('input', help="Input Jazelle file path")
    p_inspect.add_argument('-n', '--lines', type=int, default=10, help="Number of events to display")
    
    group = p_inspect.add_mutually_exclusive_group()
    group.add_argument('--head', action='store_true', default=True, help="Show first N events")
    group.add_argument('--tail', action='store_true', help="Show last N events")
    
    p_inspect.add_argument('-b', '--banks', nargs='+', help="Specific bank counts to show in table (e.g. 'MC*', 'PHCHRG')")
    p_inspect.add_argument('-t', '--threads', type=int, default=None, help="Number of threads (0=auto)")
    
    p_inspect.add_argument(
        '-d', '--display-options', 
        type=str, 
        help='Display configuration (e.g. "max_rows=5,show_dimensions=0,show_arrays=0")'
    )
    
    p_inspect.set_defaults(func=handle_inspect)

    # --- READ Command ---
    p_read = subparsers.add_parser(
        'read', 
        help="Read content of a specific event",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p_read.add_argument('input', help="Input Jazelle file path")
    p_read.add_argument('-i', '--index', type=int, default=0, help="Event index to read")
    p_read.add_argument('-b', '--banks', nargs='+', help="Bank filters (supports wildcards, e.g. 'MC*', '*')")
    p_read.add_argument('-l', '--limit', type=int, default=0, help="Limit output lines (0 = unlimited)")
    
    p_read.add_argument(
        '-d', '--display-options', 
        type=str, 
        help='Display configuration (e.g. "max_rows=50,float_precision=2")'
    )
    
    p_read.set_defaults(func=handle_read)

    # --- CONVERT Command ---
    p_convert = subparsers.add_parser(
        'convert', 
        help="Convert Jazelle file to other formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p_convert.add_argument('-i', '--input', required=True, help="Input Jazelle file")
    p_convert.add_argument('-o', '--output', required=True, help="Output file (format inferred from extension: .parquet, .h5, .json)")
    p_convert.add_argument('--start', type=int, default=0, help="Start event index")
    p_convert.add_argument('--count', type=int, default=-1, help="Number of events (-1 for all)")
    p_convert.add_argument('--batch-size', type=int, default=1000, help="Read batch size")
    p_convert.add_argument('-t', '--threads', type=int, default=None, help="Number of threads (0=auto)")
    p_convert.set_defaults(func=handle_convert)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == "__main__":
    main()