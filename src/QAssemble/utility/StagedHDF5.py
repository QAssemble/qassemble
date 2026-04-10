import hashlib
import json
import logging
import os
import re
import shutil
from typing import Dict, List, Optional, Set, Tuple

import h5py
import numpy as np

logger = logging.getLogger("QAssemble")

_LOC_KEY_TO_COMM_KEY = {
    "kloc2glob": "commk",
    "rloc2glob": "commk",
    "floc": "commfermion",
    "bloc": "commboson",
}


def _sanitize_component(component: str) -> str:

    text = str(component)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "root"


def _require_group(parent, path: str):

    if not path:
        return parent

    group = parent
    for name in str(path).split("/"):
        if name:
            group = group.require_group(name)

    return group


def _write_dataset(
    hdf5file: str,
    group: str,
    subgroup: str,
    dataset_name: str,
    data: np.ndarray,
) -> None:

    with h5py.File(hdf5file, "a") as file:
        group_obj = _require_group(file, group)
        subgroup_obj = _require_group(group_obj, subgroup)

        if dataset_name in subgroup_obj:
            del subgroup_obj[dataset_name]

        subgroup_obj.create_dataset(
            dataset_name,
            data=np.asarray(data),
            dtype=np.asarray(data).dtype,
        )


def _write_scalars(
    hdf5file: str,
    group: str,
    subgroup: str,
    scalar_datasets: Optional[dict],
) -> None:

    if not scalar_datasets:
        return

    with h5py.File(hdf5file, "a") as file:
        group_obj = _require_group(file, group)
        subgroup_obj = _require_group(group_obj, subgroup)

        for name, value in scalar_datasets.items():
            if name in subgroup_obj:
                del subgroup_obj[name]

            arr = np.asarray(value)
            subgroup_obj.create_dataset(name, data=arr, dtype=arr.dtype)


def _world_comm():

    from mpi4py import MPI

    return MPI.COMM_WORLD


def _global_size_from_loc_dict(loc_dict: dict) -> int:

    max_idx = -1
    for rank_map in loc_dict.values():
        for global_idx in rank_map.values():
            max_idx = max(max_idx, int(global_idx))

    return max_idx + 1


def _local_indices(loc_dict: dict, comm_rank: int) -> List[int]:

    rank_map = loc_dict[comm_rank]
    return [int(rank_map[i]) for i in range(len(rank_map))]


def _stage_dir(hdf5file: str, group: str, subgroup: str, dataset_name: str) -> str:

    abspath = os.path.abspath(hdf5file)
    digest = hashlib.sha1(
        f"{abspath}::{group}::{subgroup}::{dataset_name}".encode("utf-8")
    ).hexdigest()[:12]

    folder = "__".join(
        [
            _sanitize_component(group),
            _sanitize_component(subgroup),
            _sanitize_component(dataset_name),
            digest,
        ]
    )

    return os.path.join(os.path.dirname(abspath), ".staged_hdf5", folder)


def _owner_for_replicated_comms(nodedict: dict, replicated_comm_keys: Set[str]) -> bool:

    if not replicated_comm_keys:
        return True

    return all(nodedict[comm_key].Get_rank() == 0 for comm_key in replicated_comm_keys)


def _build_shard_metadata(
    data: np.ndarray,
    nodedict: dict,
    distributed_axes: Optional[List[Tuple[int, str]]],
    replicated_comm_keys: Optional[List[str]],
    world_rank: int,
) -> Tuple[dict, Set[str]]:

    axes = []
    global_shape = [int(size) for size in data.shape]
    replicated = set(replicated_comm_keys or [])

    for axis, loc_key in distributed_axes or []:
        if loc_key not in _LOC_KEY_TO_COMM_KEY:
            raise ValueError(f"Unsupported distributed axis key '{loc_key}'.")

        comm_key = _LOC_KEY_TO_COMM_KEY[loc_key]
        loc_dict = nodedict[loc_key]
        comm = nodedict[comm_key]
        comm_rank = comm.Get_rank()
        indices = _local_indices(loc_dict, comm_rank)
        local_size = len(indices)
        global_size = _global_size_from_loc_dict(loc_dict)
        axis_size = int(data.shape[axis])

        axis_meta = {
            "axis": int(axis),
            "loc_key": loc_key,
            "comm_key": comm_key,
            "comm_rank": int(comm_rank),
            "global_size": int(global_size),
            "local_size": int(local_size),
            "mode": None,
            "indices": None,
        }

        global_shape[axis] = int(global_size)

        if axis_size == local_size:
            axis_meta["mode"] = "local"
            axis_meta["indices"] = indices
        elif axis_size == global_size:
            axis_meta["mode"] = "global"
            replicated.add(comm_key)
        else:
            raise ValueError(
                f"Axis {axis} has length {axis_size}, but expected local {local_size} "
                f"or global {global_size} for loc key '{loc_key}'."
            )

        axes.append(axis_meta)

    if not axes and not replicated:
        owner = world_rank == 0
    else:
        owner = _owner_for_replicated_comms(nodedict, replicated)

    metadata = {
        "world_rank": int(world_rank),
        "owner": bool(owner),
        "dtype": np.asarray(data).dtype.str,
        "local_shape": [int(size) for size in data.shape],
        "global_shape": global_shape,
        "axes": axes,
        "replicated_comm_ranks": {
            comm_key: int(nodedict[comm_key].Get_rank()) for comm_key in sorted(replicated)
        },
    }

    return metadata, replicated


def _make_indexer(ndim: int, axis_indices: Dict[int, List[int]]) -> tuple:

    axes = sorted(axis_indices)

    if not axes:
        return (Ellipsis,)

    indexer = [slice(None)] * ndim
    mesh = np.ix_(*(np.array(axis_indices[axis], dtype=int) for axis in axes))

    for axis, values in zip(axes, mesh):
        indexer[axis] = values

    return tuple(indexer)


def _consolidate_from_manifest(manifest_path: str) -> None:

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    dtype = np.dtype(manifest["dtype"])
    global_shape = tuple(manifest["global_shape"])
    dataset = np.zeros(global_shape, dtype=dtype)

    owner_shards = [shard for shard in manifest["shards"] if shard["owner"]]

    if not owner_shards:
        raise RuntimeError(
            f"No owner shards were found for dataset '{manifest['dataset_name']}'."
        )

    for shard in owner_shards:
        shard_path = shard["path"]
        local_shape = tuple(shard["local_shape"])
        expected_size = int(np.prod(local_shape, dtype=np.int64))

        with open(shard_path, "rb") as handle:
            shard_data = np.fromfile(handle, dtype=dtype)

        if shard_data.size != expected_size:
            raise ValueError(
                f"Shard '{shard_path}' has {shard_data.size} elements, "
                f"expected {expected_size}."
            )

        shard_data = shard_data.reshape(local_shape)

        axis_indices = {
            int(axis_meta["axis"]): axis_meta["indices"]
            for axis_meta in shard["axes"]
            if axis_meta["mode"] == "local"
        }

        if axis_indices:
            dataset[_make_indexer(len(global_shape), axis_indices)] = shard_data
        else:
            dataset[...] = shard_data

    _write_dataset(
        manifest["hdf5file"],
        manifest["group"],
        manifest["subgroup"],
        manifest["dataset_name"],
        dataset,
    )


def save_distributed_dataset(
    *,
    hdf5file: str,
    group: str,
    subgroup: str,
    dataset_name: str,
    data: np.ndarray,
    nodedict: Optional[dict] = None,
    distributed_axes: Optional[List[Tuple[int, str]]] = None,
    replicated_comm_keys: Optional[List[str]] = None,
    scalar_datasets: Optional[dict] = None,
) -> None:

    array = np.asarray(data)

    if nodedict is None:
        _write_dataset(hdf5file, group, subgroup, dataset_name, array)
        _write_scalars(hdf5file, group, subgroup, scalar_datasets)
        return

    world = _world_comm()
    world_rank = world.Get_rank()

    stage_dir = _stage_dir(hdf5file, group, subgroup, dataset_name)
    manifest_path = os.path.join(stage_dir, "manifest.json")
    shard_path = os.path.join(stage_dir, f"rank_{world_rank:05d}.bin")

    metadata, replicated = _build_shard_metadata(
        array,
        nodedict,
        distributed_axes,
        replicated_comm_keys,
        world_rank,
    )

    if world_rank == 0:
        if os.path.isdir(stage_dir):
            shutil.rmtree(stage_dir)
        os.makedirs(stage_dir, exist_ok=True)

    world.Barrier()

    if metadata["owner"]:
        with open(shard_path, "wb") as handle:
            np.ascontiguousarray(array).tofile(handle)
        metadata["path"] = shard_path
    else:
        metadata["path"] = None

    gathered = world.gather(metadata, root=0)

    if world_rank == 0:
        manifest = {
            "hdf5file": hdf5file,
            "group": group,
            "subgroup": subgroup,
            "dataset_name": dataset_name,
            "dtype": array.dtype.str,
            "global_shape": metadata["global_shape"],
            "distributed_axes": distributed_axes or [],
            "replicated_comm_keys": sorted(replicated),
            "shards": gathered,
        }

        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        logger.info(
            "Consolidating staged dataset '%s/%s/%s' from %d owner shards.",
            group,
            subgroup,
            dataset_name,
            sum(1 for shard in gathered if shard["owner"]),
        )
        _consolidate_from_manifest(manifest_path)
        _write_scalars(hdf5file, group, subgroup, scalar_datasets)
        shutil.rmtree(stage_dir)

    world.Barrier()
