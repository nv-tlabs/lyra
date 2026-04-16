from hydra.core.config_store import ConfigStore
from lyra_2._src.datasets.depth_warp_dataloader import get_gen3c_multiple_video_dataloader


def lyra_register_dataloaders():
    """Register lyra_2 dataloaders."""
    cs = ConfigStore.instance()

    lyra2_dl3dv_long_480p_dav3_hsg = get_gen3c_multiple_video_dataloader(
        dataset_list=["dl3dv_long_moge_chunk_81_480p_dav3_hsg"],
        dataset_weight_list=[1],
        num_workers=2,
        prefetch_factor=2,
    )
    cs.store(group="data_train", package="dataloader_train", name="lyra2_dl3dv_long_moge_chunk_81_480p_dav3_hsg", node=lyra2_dl3dv_long_480p_dav3_hsg)
    cs.store(group="data_val", package="dataloader_val", name="lyra2_dl3dv_long_moge_chunk_81_480p_dav3_hsg", node=lyra2_dl3dv_long_480p_dav3_hsg)
