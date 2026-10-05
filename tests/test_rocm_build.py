"""The RDNA extension name follows its sources, headers and flags, so a header change rebuilds (any machine)."""

from tensorfold.rocm.kernels.build import _digest


def test_a_header_change_alone_changes_the_build(tmp_path):
    source = tmp_path / "kernel.hip"
    header = tmp_path / "layout.hpp"
    source.write_text('#include "layout.hpp"\n')
    header.write_text("struct A { float* p; };\n")
    flags = {"extra_cuda_cflags": ["-O3"], "extra_include_paths": [str(tmp_path)]}
    before = _digest([str(source)], flags)
    assert _digest([str(source)], flags) == before
    header.write_text("struct A { const void* p; int kind; };\n")
    assert _digest([str(source)], flags) != before


def test_flags_change_the_build(tmp_path):
    source = tmp_path / "kernel.hip"
    source.write_text("\n")
    one = _digest([str(source)], {"extra_cuda_cflags": ["-DTENSORFOLD_RDNA_WMMA=1"]})
    assert one != _digest([str(source)], {"extra_cuda_cflags": ["-DTENSORFOLD_RDNA_WMMA=0"]})
    assert one == _digest([str(source)], {"extra_cuda_cflags": ["-DTENSORFOLD_RDNA_WMMA=1"], "verbose": True})
