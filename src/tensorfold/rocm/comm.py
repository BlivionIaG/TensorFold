"""RCCL all-gather / all-reduce on the current HIP stream so HIP graphs capture it.

Every rank receives the same ``all_reduce`` result; ``ready`` and ``barrier`` are the
rendezvous that the follower rank uses to wait for the leader. Same shape as ``tensorfold.cuda.comm``
because librccl exports NCCL's C API under NCCL's names (ncclGetUniqueId, ncclCommInitRank, ncclAllReduce, ...).

Hardware support:
- RDNA 3.5 / RDNA 4 integrated (``multi_gpu_capable=True``): dies on one APU package; P2P over
  package-local HBM; the user may set ``prefer_p2p=True`` and the engine turns on ``RCCL_USE_P2P=1``.
- RDNA 2 / RDNA 3 / RDNA 4 discrete (multi-card via motherboard): PCIe P2P is BIOS-, ACS- and
  driver-dependent. ``prefer_p2p`` defaults to ``False``; the engine refuses with a clear error if
  the user passes ``--p2p`` and ``hipDeviceCanAccessPeer`` returns 0 for the pair.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os

import torch

_DTYPES = {
    torch.float32: 7,    # ncclFloat32
    torch.float16: 6,    # ncclFloat16
    torch.bfloat16: 9,   # ncclBfloat16
    torch.int32: 2,      # ncclInt32
    torch.int64: 4,      # ncclInt64
}
_OPS = {"sum": 0, "prod": 1, "max": 2, "min": 3}


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library() -> ctypes.CDLL:
    """Find ``librccl.so`` (or override via ``TF_RCCL_LIB``); refuse with a clear error otherwise."""

    if os.name == "nt":
        raise RuntimeError("TensorFold does not run tensor-parallel (RCCL) on Windows")
    candidates = [os.environ.get("TF_RCCL_LIB", "")]
    found = ctypes.util.find_library("rccl")
    if found:
        candidates.append(found)
    candidates += (
        glob.glob("/opt/rocm/lib/librccl*.so*")
        + glob.glob("/usr/lib/x86_64-linux-gnu/librccl*.so*")
        + glob.glob("/usr/local/lib/librccl*.so*")
        + glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "librccl*.so*"))
    )
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError("librccl not found (set TF_RCCL_LIB or `apt install rccl`)")


class RCCL:
    """One rank of an RCCL ring. Stream-aware so HIP-graph capture works.

    Args:
        rank: this process's rank in ``[0, world)``.
        world: total ranks (``tp``); 1 disables multi-rank and is not constructed here.
        master: rank-0's reachable address (used by ``torch.distributed.TCPStore``).
        port: rendezvous port (default 29551; matches ``--master-port``).
        prefer_p2p: when ``True``, set ``RCCL_USE_P2P=1``; when ``False``, force host-memory staging.
            Default ``False``; APU multi-die devices set it from the engine (RDNA 3.5/4 integrated with
            ``multi_gpu_capable=True``). Disagreeing with ``hipDeviceCanAccessPeer`` is an explicit
            error at startup, never a silent downgrade.
    """

    def __init__(self, rank: int, world: int, master: str, port: int, *, prefer_p2p: bool = False) -> None:
        if world < 2:
            raise ValueError("RCCL is for multi-rank: tp=1 needs no comm")
        if not (0 <= rank < world):
            raise ValueError(f"rank {rank} not in [0, {world})")
        if not master:
            raise ValueError("--master: rank 0's address on the link between the machines")

        from datetime import timedelta

        from torch.distributed import TCPStore

        self.rank, self.world, self.prefer_p2p = rank, world, bool(prefer_p2p)
        self.lib = _library()
        lib = self.lib
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int,
        ]
        lib.ncclAllGather.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.ncclAllReduce.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.ncclBroadcast.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]

        self.store = TCPStore(master, port, world, rank == 0, timeout=timedelta(seconds=600))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set("tf_rccl_uid", bytes(uid.internal))
        else:
            raw = self.store.get("tf_rccl_uid")
            ctypes.memmove(ctypes.addressof(uid), raw, 128)

        os.environ["RCCL_USE_P2P"] = "1" if self.prefer_p2p else "0"

        self.comm = ctypes.c_void_p()
        torch.cuda.current_device()
        self._check(self.lib.ncclCommInitRank(
            ctypes.byref(self.comm), ctypes.c_int(world), uid, ctypes.c_int(rank),
        ))

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"RCCL error {code}: {self.lib.ncclGetErrorString(code).decode()}")

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], contiguous on this stream so HIP-graph capture keeps it."""
        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclAllGather(
            send.data_ptr(), recv.data_ptr(), send.numel(), _DTYPES[send.dtype], self.comm, stream,
        ))

    def all_reduce(self, send: torch.Tensor, recv: torch.Tensor, *, op: str = "sum") -> None:
        """``send`` summed (or ``op``) across ranks into ``recv``, on this stream. Same shape and dtype.

        The ring decides the order of the adds. With two ranks that is one add; past two, fp32 sums can
        differ in the last bit from a rank-order sum.
        """
        if op not in _OPS:
            raise ValueError(f"all_reduce: op must be one of {list(_OPS)}, not {op!r}")
        if send.dtype != recv.dtype or send.numel() != recv.numel():
            raise ValueError("all_reduce: send and recv must share shape and dtype")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclAllReduce(
            send.data_ptr(), recv.data_ptr(), send.numel(),
            _DTYPES[send.dtype], _OPS[op], self.comm, stream,
        ))

    def broadcast(self, send: torch.Tensor, recv: torch.Tensor, *, root: int) -> None:
        """Stream-aware broadcast: ``root`` sends, every rank (including root) receives into ``recv``."""

        if not (0 <= root < self.world):
            raise ValueError(f"broadcast: root {root} not in [0, {self.world})")
        if send.dtype != recv.dtype or send.numel() != recv.numel():
            raise ValueError("broadcast: send and recv must share shape and dtype")
        stream = torch.cuda.current_stream().cuda_stream
        self._check(self.lib.ncclBroadcast(
            send.data_ptr(), recv.data_ptr(), send.numel(),
            _DTYPES[send.dtype], ctypes.c_int(root), self.comm, stream,
        ))

    def ready(self, label: str, *, every: float = 60.0, timeout: float = 3600.0) -> None:
        """Every rank finishes ``label`` before any goes on; a rank missing after ``timeout`` s is named."""
        import time
        from datetime import timedelta

        self.store.set(f"tf_ready/{label}/{self.rank}", "1")
        others = [r for r in range(self.world) if r != self.rank]
        started = time.monotonic()
        while True:
            try:
                self.store.wait([f"tf_ready/{label}/{r}" for r in others], timedelta(seconds=every))
                return
            except Exception as exc:
                if "timeout" not in str(exc).lower():
                    raise
            waited = time.monotonic() - started
            missing = ", ".join(str(r) for r in others)
            if waited >= timeout:
                raise RuntimeError(
                    f"rank {self.rank} finished {label} but rank {missing} has not after "
                    f"{waited / 60:.0f} min: check that rank's log (an HIP extension build waiting on a "
                    "lock names the lock there)"
                )
            print(
                f"[tensorfold] rank {self.rank} finished {label}; waiting for rank {missing} ({waited:.0f} s)",
                flush=True,
            )

    def barrier(self) -> None:
        """Stream-aware barrier (small all-gather + sync)."""
        x = torch.zeros((1,), dtype=torch.float32, device="cuda")
        y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
        self.all_gather(x, y)
        torch.cuda.synchronize()


__all__ = ["RCCL", "gather_ints"]


def gather_ints(rccl: RCCL | None, torch, values: list[int]) -> list[list[int]]:
    """Concatenate ``values`` across ranks in rank order; rccl=None keeps the local list (tp=1).

    Same shape as ``tensorfold.cuda.comm.gather_ints`` so the per-rank capacity module can be
    shared between backends: the gather callable abstracts RCCL / NCCL / no-op.
    """
    if rccl is None:
        return [list(values)]
    send = torch.tensor(values, dtype=torch.int64, device="cuda")
    receive = torch.empty((rccl.world * len(values),), dtype=torch.int64, device="cuda")
    rccl.all_gather(send, receive)
    return receive.view(rccl.world, -1).tolist()