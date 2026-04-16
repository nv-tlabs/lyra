from hydra.core.config_store import ConfigStore
from lyra_2._ext.imaginaire.lazy_config import LazyCall as L
from lyra_2._src.models.lyra2_model import (
    Lyra2Model,
    Lyra2T2VConfig,
)


fsdp_wan2pt1_lyra2_spatial_config = dict(
    trainer=dict(distributed_parallelism="fsdp"),
    model=L(Lyra2Model)(
        config=Lyra2T2VConfig(fsdp_shard_size=8, state_t=20),
        _recursive_=False,
    ),
)

ddp_wan2pt1_lyra2_spatial_config = dict(
    trainer=dict(distributed_parallelism="ddp"),
    model=L(Lyra2Model)(
        config=Lyra2T2VConfig(state_t=20),
        _recursive_=False,
    ),
)

def lyra_register_model():
    cs = ConfigStore.instance()
    cs.store(group="model", package="_global_", name="fsdp_wan2pt1_lyra2_spatial", node=fsdp_wan2pt1_lyra2_spatial_config)
    cs.store(group="model", package="_global_", name="ddp_wan2pt1_lyra2_spatial", node=ddp_wan2pt1_lyra2_spatial_config)
