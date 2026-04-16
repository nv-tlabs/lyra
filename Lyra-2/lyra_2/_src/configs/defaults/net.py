from hydra.core.config_store import ConfigStore
from lyra_2._ext.imaginaire.lazy_config import LazyCall as L
from lyra_2._ext.imaginaire.lazy_config import LazyDict
from lyra_2._src.networks.wan2pt1_lyra2 import Lyra2WanModel
from lyra_2._src.modules.selective_activation_checkpoint import SACConfig


WAN2PT1_1PT3B_I2V_LYRA2: LazyDict = L(Lyra2WanModel)(
    dim=1536,
    eps=1e-06,
    ffn_dim=8960,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=12,
    num_layers=30,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="block_wise"),
    postpone_checkpoint=False,
)

WAN2PT1_14B_I2V_LYRA2: LazyDict = L(Lyra2WanModel)(
    dim=5120,
    eps=1e-06,
    ffn_dim=13824,
    freq_dim=256,
    in_dim=36,
    model_type="i2v",
    num_heads=40,
    num_layers=40,
    out_dim=16,
    text_len=512,
    cp_comm_type="p2p",
    sac_config=L(SACConfig)(mode="block_wise"),
    postpone_checkpoint=False,
)


def lyra_register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="wan2pt1_1pt3B_i2v_lyra2", node=WAN2PT1_1PT3B_I2V_LYRA2)
    cs.store(group="net", package="model.config.net", name="wan2pt1_14B_i2v_lyra2", node=WAN2PT1_14B_I2V_LYRA2)
