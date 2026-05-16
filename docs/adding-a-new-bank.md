# Adding a new bank family

This guide walks through everything you need to touch to add a new bank
family to the Jazelle reader. We use **MCPNT** as the running example.

A "bank family" is a homogeneous collection of records (one per
`MCPNT`, one per `MCPART`, etc.) parsed out of the MINIDST data
buffer. Each family corresponds to one Fortran-era bank template and
one entry in the PHMTOC table of contents.

## Checklist

For a new bank `XYZAB`:

1. [C++ bank type]      Create `include/jazelle/banks/XYZAB.hpp` and
                        `src/banks/XYZAB.cpp`
2. [C++ build]          Add `src/banks/XYZAB.cpp` to `CMakeLists.txt`
3. [C++ registration]   Add to `include/jazelle/AllBanks.hpp`
4. [C++ registration]   Add to `include/jazelle/BankTraits.hpp`
5. [C++ registration]   Add to the `BankTypes` list in
                        `include/jazelle/JazelleEvent.hpp`
6. [C++ registration]   Add to `kKnownFamilies` in
                        `src/JazelleFile.cpp`
7. [C++ parser]         Add the read loop to `parseMiniDst` in
                        `src/JazelleFile.cpp` (plus parent/pointer
                        resolution if applicable)
8. [Python bindings]    Add the `cppclass` block to
                        `bindings/jazelle_cython.pxd`
9. [Python bindings]    Add the `cdef class` wrapper in
                        `bindings/jazelle_cython.pyx`
10. [Python bindings]   Add the family accessor property to the
                        `JazelleEvent` class in `jazelle_cython.pyx`
11. [Python bindings]   Register the bank's extractor in
                        `_init_extractors()` in `jazelle_cython.pyx`
12. [Python package]    Re-export from `jazelle_reader/__init__.py`

The order matters: steps 8–10 will fail to compile until 1–6 are in
place, and step 7 won't actually populate the family until step 5 makes
it discoverable to the event.

## 1. The bank type

### `include/jazelle/banks/XYZAB.hpp`

```cpp
#pragma once

#include "../Bank.hpp"
#include <cstdint>

namespace jazelle
{
    class DataBuffer;
    class JazelleEvent;

    /**
     * @struct XYZAB
     * @brief One-line description of what this bank represents.
     */
    struct XYZAB : public Bank
    {
        // Bit-flag definitions, if the template defines any:
        struct SomeBitField {
            static constexpr int32_t FOO = (1 << 0);
            static constexpr int32_t BAR = (1 << 1);
        };

        // Plain-old-data members (one per template field). Prefer
        // scalar floats over std::array<float, N> so the Cython
        // bindings can expose each as a flat property — see MCPART for
        // the rationale.
        float    px, py, pz;
        int32_t  some_int;
        // ... etc.

        // Pointers to related banks resolved after the family is built
        // (see JazelleFile.cpp's resolution loop):
        int32_t  parent_id;
        XYZAB*   parent = nullptr;

        explicit XYZAB(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset,
                     JazelleEvent& event) override;
    };
} // namespace jazelle
```

### `src/banks/XYZAB.cpp`

```cpp
#include "jazelle/banks/XYZAB.hpp"
#include "DataBuffer.hpp"

namespace jazelle
{
    int32_t XYZAB::read(const DataBuffer& buffer, int32_t offset,
                        JazelleEvent& /*event*/)
    {
        int32_t o = offset;

        px       = buffer.readFloat(o);  o += 4;
        py       = buffer.readFloat(o);  o += 4;
        pz       = buffer.readFloat(o);  o += 4;
        some_int = buffer.readInt(o);    o += 4;
        // ... etc.

        return o - offset;   // total bytes consumed
    }
} // namespace jazelle
```

**Conventions:**

- Return the number of bytes consumed. The caller (`parseMiniDst`)
  uses this to advance its `offset` cursor.
- If the bank has a variable-length encoding (e.g. PHPOINT-style
  mask-driven pointers, or MCPART's optional E field), the read
  function chooses the size dynamically — return the actual count.
- Keep the bank's POD members public; the Cython bindings read them
  directly without an accessor layer.
- `m_id` (the bank ID) is inherited from `Bank` and can be set by
  the caller via `add(id)`, or written directly inside `read()` if
  the ID is packed into the body. See MCPART for an example of the
  latter.

## 2. CMakeLists.txt

The bank source file must be added to the build. Find the list of bank
sources in `CMakeLists.txt` (look for the existing entries like
`src/banks/MCPART.cpp`, `src/banks/PHPOINT.cpp`, etc.) and add yours
alphabetically:

```cmake
# Bank implementations
set(JAZELLE_BANK_SOURCES
    src/banks/IEVENTH.cpp
    src/banks/MCHEAD.cpp
    src/banks/MCPART.cpp
    src/banks/MCPNT.cpp
    # ...
    src/banks/XYZAB.cpp     # <-- add here
    # ...
)
```

The exact variable name and form depends on how the build is
structured — if there's no list and bank sources are globbed
(`file(GLOB ...)`), no change is needed, but explicit lists are
preferred since they fail loudly when files are missing.

A clean rebuild (`rm -rf build/ && cmake -B build && cmake --build
build`) is recommended after CMake list changes; some generators don't
pick up new sources on incremental builds.

## 3. AllBanks.hpp

Add `#include "banks/XYZAB.hpp"` next to the other bank includes.
This lets downstream code pull in every bank type with a single
include.

## 4. BankTraits.hpp

Add the bank-name specialization:

```cpp
template<> inline constexpr std::string_view bank_name<XYZAB> = "XYZAB";
```

This is used by `Family::name()`, `to_dict`, table display, etc. The
string must match how the bank is referred to in the TOC documentation
and in user-facing output.

## 5. JazelleEvent.hpp

Add `XYZAB` to the `BankTypes` typelist (or equivalent — the exact
form depends on whether the event uses a `std::tuple<Family<T>...>`,
a manual `std::variant`, or a hand-rolled list). The typelist defines
which families an event holds and is the source of truth for
`event.get<XYZAB>()`.

## 6. kKnownFamilies in JazelleFile.cpp

`kKnownFamilies` is the runtime registry that lets the reader iterate
over all bank families by name (e.g. for the offsets map and the
display methods). It's typically a `std::array` or initializer list at
file scope near the top of `JazelleFile.cpp`:

```cpp
static constexpr std::array kKnownFamilies = {
    "IEVENTH",
    "MCHEAD",
    "MCPART",
    "MCPNT",
    // ...
    "XYZAB",     // <-- add here, in binary-stream order
    // ...
};
```

Order matters here — entries should appear in the same order the
families appear in the binary stream, since downstream tooling (e.g.
`getBankFamilyOffset(name)`) relies on the registry order to produce
sensible diagnostics. Forgetting this step won't cause parse errors,
but the bank won't show up in `getBankFamilyOffsets()`, `display()`,
or any name-driven introspection.

## 7. JazelleFile.cpp — parseMiniDst

In `parseMiniDst`, between the families that come before and after
XYZAB in the binary stream, add the read loop. The minimal form:

```cpp
RECORD_FAMILY_OFFSET("XYZAB");
auto& xyzabFam = event.get<XYZAB>();
xyzabFam.reserve(toc.m_nXyzab);   // count comes from PHMTOC

for (int32_t i = 0; i < toc.m_nXyzab; i++) {
    // If the bank ID is the first 2 or 4 bytes of the row, peek it:
    const int32_t bank_id = buffer.readShort(offset);  // or readInt(offset+N)
    XYZAB* b = xyzabFam.add(bank_id);
    offset += b->read(buffer, offset, event);
}
```

**Variations:**

- **ID packed inside the body**: peek the ID-word at the byte offset
  where it lives. MCPART does this at `offset + 12`.
- **Variable-length**: just call `read()` — it returns the actual
  size. The cursor advances correctly even when rows differ.
- **Multi-pass banks** (two-keyed, like MCPNT): write each pass in
  sequence. Pass 2 typically just fills in pointer fields on banks
  already created in Pass 1.

After all bank families are read, resolve cross-family pointers in
a separate loop:

```cpp
for (size_t i = 0; i < xyzabFam.size(); ++i) {
    XYZAB* b = xyzabFam.at(i);
    b->parent = (b->parent_id > 0)
                ? xyzabFam.find(b->parent_id)
                : nullptr;
}
```

## 8. Cython bindings — `bindings/jazelle_cython.pxd`

Add the C++ class declaration:

```cython
cdef extern from "jazelle/banks/XYZAB.hpp" namespace "jazelle":
    cdef cppclass CppXYZAB "jazelle::XYZAB"(CppBank):
        CppXYZAB(int32_t)
        float px, py, pz
        int32_t some_int, parent_id
```

Only declare the members Python needs to read; methods and helpers
can be omitted.

## 9. Cython bindings — `bindings/jazelle_cython.pyx`

Add a `cdef class XYZAB(Bank)` with:

- `__init__` that refuses direct instantiation
- `__repr__`
- One `@property` per scalar field
- A `cpdef dict to_dict(self)` returning a flat dict
- A `@staticmethod def bulk_extract(Family family)` returning a dict
  of 1-D numpy arrays
- A `@staticmethod cdef dict extract_from_vector(...)` used by the
  parallel batch path

Use the **MCPART** wrapper as the template — it's the most fleshed-out
example and covers all the patterns. Do **not** expose 3-vector arrays
(`p`, `xt`); break them into scalar properties (`px, py, pz`,
`xt_x, xt_y, xt_z`) for consistency with PHPSUM.

## 10. JazelleEvent family accessor (same `.pyx` file)

For each bank family there's a snake-case `@property` on `JazelleEvent`
that returns the wrapped family directly, so users can write
`event.xyzab` instead of `event.getFamily("XYZAB")`. Find the cluster
of existing accessors near the top of the class (look for
`def mcpart(self):`) and add yours in binary-stream order:

```cython
cdef class JazelleEvent:
    # ... existing fields ...

    @property
    def mcpart(self):
        return wrap_family(&self.cpp_event.get[pxd.CppMCPART](),
                           self, MCPART)

    @property
    def xyzab(self):                                           # <-- add
        return wrap_family(&self.cpp_event.get[pxd.CppXYZAB](),
                           self, XYZAB)

    # ... other existing accessors ...
```

The property's name should be lowercase (`xyzab`, `mcpnt`, `phwmc`)
to match Python convention; the C++ template argument (`CppXYZAB`)
and the Python wrapper class (`XYZAB`) keep their canonical casing.

Forgetting this step doesn't break `event.getFamily("XYZAB")` — that
path goes through `_WRAPPER_MAP` and works as long as steps 1–9 are
done. What you lose is the convenient attribute-style access and IDE
autocomplete.

## 11. Register the extractor in `_init_extractors()` (same `.pyx` file)

`_init_extractors()` is the single source of truth for which banks
participate in the parallel batch extraction path. It maps a bank
name to its `extract_from_vector` static method. Without registration,
`bulk_extract` for that bank works via the per-family path but the
batch / parallel path silently skips it.

Find `_init_extractors` near the bottom of the file and add the
registration:

```cython
cdef void _init_extractors():
    """
    Populate the C++ function pointer map.
    This runs once when the module is imported.
    """
    register_extractor(b"IEVENTH", IEVENTH.extract_from_vector)
    register_extractor(b"MCHEAD",  MCHEAD.extract_from_vector)
    register_extractor(b"MCPART",  MCPART.extract_from_vector)
    # ...
    register_extractor(b"XYZAB",   XYZAB.extract_from_vector)   # <-- add
    # ...
```

The byte-string name (`b"XYZAB"`) must match the canonical bank name
used everywhere else — same as `BankTraits.hpp`, same as
`kKnownFamilies`, same as the `_WRAPPER_MAP` key. If the strings don't
match, the batch path silently returns nothing for that family.

## 12. Python package — `jazelle_reader/__init__.py`

The Python wrapper class needs to be re-exported from the package's
top-level namespace so users can write `from jazelle_reader import
XYZAB` (or `jazelle_reader.XYZAB`). Add it to the import line that
pulls bank classes from the compiled extension:

```python
from .jazelle_cython import (
    JazelleFile,
    JazelleEvent,
    Family,
    Bank,
    IEVENTH, MCHEAD, MCPART, MCPNT,
    # ...
    XYZAB,                # <-- add here
    # ...
)
```

If the file uses an `__all__` list (it should, for `from
jazelle_reader import *` to be well-defined), add `"XYZAB"` there
too:

```python
__all__ = [
    "JazelleFile", "JazelleEvent", "Family", "Bank",
    "IEVENTH", "MCHEAD", "MCPART", "MCPNT",
    # ...
    "XYZAB",
    # ...
]
```

Forgetting this step doesn't break parsing — the bank is still in
`event.get_family("XYZAB")` — but `from jazelle_reader import XYZAB`
will raise ImportError and IDE autocomplete won't find it.

## Rebuild

```bash
# Clean build is safest after CMake/typelist changes:
rm -rf build/
cmake -B build
cmake --build build

# Rebuild the .so the python package imports:
pip install -e .
```

Common failure modes after partial rebuilds:

- **Stale `.o` files** — symptoms include "duplicate bank id" runtime
  errors that don't match the new logic, or family sizes off by one.
  Fix: `rm -rf build/` and rebuild from scratch.
- **Cython `.so` not rebuilt** — symptoms include ImportError or the
  new bank class not appearing in the module. Fix: re-run
  `pip install -e .`.
- **Bank parses but `event.xyzab` raises AttributeError** — step 10
  (JazelleEvent property) was skipped. The bank is in the event's
  family tuple but lacks the named accessor. Use
  `event.getFamily("XYZAB")` as a workaround until the property is
  added.
- **Bank parses but batch extraction returns nothing for it** —
  step 11 (`_init_extractors` registration) was skipped. Per-event
  `family.bulk_extract()` still works; the parallel batch path
  doesn't.

## Sanity checks

After the rebuild, run on a known-good file:

1. **TOC counts match family sizes**:
   `assert event.get_family("XYZAB").size() == toc.n_xyzab`
2. **Family is in the known-families registry**:
   `assert "XYZAB" in file.getBankFamilyOffsets()`
3. **`offset` reaches the next family cleanly**: the loop for the
   *next* bank after XYZAB must not crash with "duplicate id" or
   out-of-range pointer errors — those are misalignment signatures.
4. **Cross-family pointers resolve**: spot-check a few `parent_id`s
   against the resolved `parent` pointer's id.
5. **Python access works**: `to_dict()`, `bulk_extract()`, and
   `__repr__` on a sample bank all run without exceptions.
6. **Top-level import works**:
   `python -c "from jazelle_reader import XYZAB; print(XYZAB)"`
7. **Attribute-style family access works**:
   `event.xyzab` returns a Family object (not an AttributeError).
8. **Batch extraction includes the bank**: when running the parallel
   batch path on a sample with `n_xyzab > 0`, the resulting dict
   contains an `'XYZAB'` key with non-empty arrays.