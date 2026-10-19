"""Reintroduce the old epilogue NaN loss in an isolated GPU proof tree only."""
from pathlib import Path
p = Path('zig/kernels/hip/tiles/epilogue.hpp')
s = p.read_text()
a = 'u = u != u ? u : fmaxf(u, 0.f);'
b = 'u = fmaxf(u, 0.f);'
assert s.count(a) == 1, 'expected exactly the fixed StreamRelu2 expression'
p.write_text(s.replace(a, b))
