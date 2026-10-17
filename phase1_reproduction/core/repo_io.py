"""Read the ROS repository's pretrained policies and stored results without PyTorch.

The repository (github.com/uoe-agents/robust_onpolicy_data_collection, commit 056cebc) is expected
at <project root>/external/robust_onpolicy_data_collection, or wherever $ROS_REPO points.
Both readers use restricted unpicklers, so no code stored in those files is ever executed.
"""
import codecs
import collections
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np

try:
    from numpy._core import multiarray      # NumPy >= 2
except ImportError:
    from numpy.core import multiarray       # NumPy 1.x

REPO = Path(os.environ.get(
    "ROS_REPO", Path(__file__).resolve().parents[2] / "external" / "robust_onpolicy_data_collection"))

# Plain containers and NumPy arrays: everything the stored files need, nothing executable.
_SAFE = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("collections", "defaultdict"): collections.defaultdict,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): multiarray.scalar,
    ("numpy._core.multiarray", "_reconstruct"): multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): multiarray.scalar,
    ("_codecs", "encode"): codecs.encode,
}
_TORCH_STORAGES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
                   "LongStorage": np.int64, "IntStorage": np.int32}


class _Inert:
    """Stand-in for any class a torch pickle refers to (e.g. the repo's Actor module)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _rebuild_tensor(storage, offset, size, stride, *_):
    item = storage.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * item for s in stride)).copy()


class _TorchUnpickler(pickle.Unpickler):
    """Unpickles a torch.save() archive into NumPy arrays and inert objects."""

    def __init__(self, fileobj, archive, prefix):
        super().__init__(fileobj)
        self._archive, self._prefix = archive, prefix

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch._utils" and name.startswith("_rebuild_parameter"):
            return lambda data, *_: data
        if module == "torch" and name in _TORCH_STORAGES:
            return name
        if (module, name) in _SAFE:
            return _SAFE[(module, name)]
        return type(name, (_Inert,), {})

    def persistent_load(self, pid):
        _, storage_type, key = pid[:3]
        dtype = _TORCH_STORAGES[storage_type if isinstance(storage_type, str) else storage_type.__name__]
        raw = self._archive.read(f"{self._prefix}/data/{key}")
        return np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"))


def _arrays(obj, seen=None):
    """Every NumPy array reachable from obj."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple)):
        children = obj
    elif isinstance(obj, _Inert):
        children = obj.__dict__.values()
    else:
        return []
    return [a for c in children for a in _arrays(c, seen)]


def policy_logits(name):
    """Logit table (n_states, n_actions) of the repo policy policy_models/model_<name>.pt."""
    with zipfile.ZipFile(REPO / "policy_models" / f"model_{name}.pt") as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        with zf.open(pkl) as f:
            module = _TorchUnpickler(f, zf, pkl[: -len("/data.pkl")]).load()
    tables = [a for a in _arrays(module) if a.ndim == 2]     # the actor's Embedding weight
    assert len(tables) == 1, [a.shape for a in tables]
    return tables[0].astype(np.float64)


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE:
            return _SAFE[(module, name)]
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def stored_run(folder, name):
    """A stored per-seed result file data/<folder>/<name> of the repo.

    Returns (estimates / v_true with shape (seeds, checkpoints), environment steps at each checkpoint).
    The repo divides every estimate by its ground-truth value, so 1.0 means exact."""
    with open(REPO / "data" / folder / name, "rb") as f:
        d = _SafeUnpickler(f).load()
    (est,) = d["estimations"].values()
    return np.array(est, dtype=float), np.array(d["eval_steps"], dtype=float)
