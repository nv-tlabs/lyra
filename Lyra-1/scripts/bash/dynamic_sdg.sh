CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py \
    --checkpoint_dir checkpoints \
    --vipe_path assets/demo/dynamic/diffusion_input/rgb/6a71ee0422ff4222884f1b2a3cba6820.mp4 \
    --video_save_folder assets/demo/dynamic/diffusion_output_generated \
    --disable_prompt_upsampler \
    --num_gpus 1 \
    --foreground_masking \
    --multi_trajectory