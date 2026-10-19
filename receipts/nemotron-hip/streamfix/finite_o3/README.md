# Production-optimization finite byte regression

This separate run preserves earlier O2 artifacts as historical evidence. Production hipcc --genco without an explicit optimization option expands to device O3; this harness makes O3 explicit and records hipcc -v expanded commands. It retains C++20, contraction off, wave32 and all production common/caps defines. The standalone host driver replaces only the production code-object loading mechanism; actual stream templates are instantiated unchanged.

The exact same finite.hip source is built against baseline 8649e985 and final cc18314d. gfx1100 executes 1080 fixtures, gfx1030 540. The comparator requires every output byte to match, including NaN payloads and signed zeros, and requires observed NaN outputs from entirely finite operands. It checks exact source commits and O3 flags; manifests retain source, binary, harness, headers, compiler flags and expanded-command log hashes.

Private raw compiler logs may contain machine paths. They are provenance inputs, not publication-ready files. No result is claimed until comparison reports and stage completion exist.
