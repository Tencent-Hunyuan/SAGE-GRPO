import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import os
import time
import random
import functools
from typing import List, Optional, Tuple, Union

from dataclasses import dataclass
from functools import cached_property


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    sp: int
    tp: int
    pp: int
    world_size: int

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, sp, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.sp,
            self.tp,
            self.pp,
        )
        for d in (dp_replicate, sp, tp, pp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * sp * tp * pp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * sp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"sp({sp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.sp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "sp", "tp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        # logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Create all the submesh here to ensure all required process groups are
        # initialized:
        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_sp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_sp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_sp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_sp_mesh_dim_names.append("dp_shard")
            dp_sp_mesh_dim_names.append("dp_shard")
        if self.sp_enabled:
            dp_shard_sp_mesh_dim_names.append("sp")
            dp_sp_mesh_dim_names.append("sp")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_sp_mesh_dim_names != []:
            mesh[tuple(dp_shard_sp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_sp"
            )
        if dp_sp_mesh_dim_names != []:
            mesh[tuple(dp_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_sp")

        self.device_mesh = mesh
        if self.sp_enabled:
            self.sp_mesh = mesh['sp']
        if self.dp_enabled:
            self.dp_mesh = mesh['dp']

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @cached_property
    def non_data_parallel_size(self):
        return self.sp * self.tp * self.pp


class COMM_INFO:
    def __init__(self):
        self.sp_group = None
        self.tp_group = None
        self.sp_size = 1
        self.tp_size = 1
        self.global_rank = 0
        self.rank_within_spgroup = 0
        self.rank_within_tpgroup = 0
        self.parallel_dims = None
        self.device_mesh = None
        self.use_dynamic_ring_attention = False
        self.sp_stream = None
        self.sp_rank_list = None
        

nccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False
_TEACHER_STUDENT_PARALLEL_STATE = False


def initialize_sequence_parallel_state_v2(parallel_dims, device_mesh, use_dynamic_ring_attention=False):
    global _SEQUENCE_PARALLEL_STATE
    if parallel_dims.sp_enabled or parallel_dims.tp_enabled:
        initialize_sequence_parallel_group_v2(parallel_dims, device_mesh, use_dynamic_ring_attention)
    else:
        nccl_info.sp_size = 1
        nccl_info.tp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_spgroup = 0
        nccl_info.rank_within_tpgroup = 0
        nccl_info.sp_group_id = int(os.getenv("RANK", "0"))
        nccl_info.tp_group_id = int(os.getenv("RANK", "0"))
        nccl_info.parallel_dims = parallel_dims
        nccl_info.device_mesh = device_mesh


def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)
    else:
        nccl_info.sp_size = 1
        nccl_info.global_rank = int(os.getenv("RANK", "0"))
        nccl_info.rank_within_group = 0
        nccl_info.group_id = int(os.getenv("RANK", "0"))


def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state

def set_tensor_parallel_state(state):
    global _TENSOR_PARALLEL_STATE
    _TENSOR_PARALLEL_STATE = state

def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE

def get_tensor_parallel_state():
    return _TENSOR_PARALLEL_STATE

def initialize_sequence_parallel_group_v2(parallel_dims, device_mesh, use_dynamic_ring_attention=False):
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    nccl_info.sp_size = parallel_dims.sp
    nccl_info.tp_size = parallel_dims.tp
    nccl_info.global_rank = rank
    nccl_info.parallel_dims = parallel_dims
    nccl_info.device_mesh = device_mesh
    nccl_info.use_dynamic_ring_attention = use_dynamic_ring_attention
    if use_dynamic_ring_attention:
        nccl_info.sp_stream = torch.cuda.Stream()
        nccl_info.sp_rank_list = device_mesh["sp"].mesh.tolist()
    if nccl_info.sp_size > 1:
        set_sequence_parallel_state(True)
        nccl_info.sp_group = device_mesh.get_group(mesh_dim="sp")
        nccl_info.rank_within_spgroup = device_mesh.get_local_rank(mesh_dim="sp")
    if nccl_info.tp_size > 1:
        set_tensor_parallel_state(True)
        nccl_info.tp_group = device_mesh.get_group(mesh_dim="tp")
        nccl_info.rank_within_tpgroup = device_mesh.get_local_rank(mesh_dim="tp")

def initialize_sequence_parallel_group(sequence_parallel_size):
    """Initialize the sequence parallel group."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert (
        world_size % sequence_parallel_size == 0
    ), "world_size must be divisible by sequence_parallel_size, but got world_size: {}, sequence_parallel_size: {}".format(
        world_size, sequence_parallel_size
    )
    nccl_info.sp_size = sequence_parallel_size
    nccl_info.global_rank = rank
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            nccl_info.group = group
            nccl_info.rank_within_group = rank - i * sequence_parallel_size
            nccl_info.group_id = i


def destroy_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    dist.destroy_process_group()