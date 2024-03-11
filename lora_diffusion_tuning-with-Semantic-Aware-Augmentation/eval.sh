export MODEL_NAME="CompVis/stable-diffusion-v1-4"
#expert TRAIN_DIR="/Data_PHD/phd22_zhaorui_tan/data/wikiart/images/vincent-van-gogh/selected"
export PYTHONDONTWRITEBYTECODE=1

CUDA_VISIBLE_DEVICES=6 accelerate launch --mixed_precision="fp16" eval.py \
  --train_data_dir="/Data_PHD/phd22_zhaorui_tan/data/wikiart/images/vincent-van-gogh/selected" \
  --output_dir="van-gogh-lora-protection-ffc-re-4" \
  --style_load=True \
  --style_path="/Data_PHD/phd22_zhaorui_tan/SDE_test/van-gogh-lora-protection-ffc-re-5/protect_final.pt" \
  --style_recon_weight=0.1 \
  --style_semantic_weight=0.1 \
#  --style_mix_ratio=0.6 \
#  --style_mix_strength=0.3 \
#  --style_maintain_strength=0.2 \ all_mse 0.008556789511265747 all_div_mse 0.011201722281319755 all_res 0.7720978098058268

#  --style_mix_ratio=0.8 \
#  --style_mix_strength=0.2 \
#  --style_maintain_strength=0.2 all_mse 0.005556796822593862 all_div_mse 0.0072618212018694195 all_res 0.7752319509389968


#  --style_mix_ratio=0.8 \
#  --style_mix_strength=0.3 \
#  --style_maintain_strength=0.2 all_mse 0.012636448972229692 all_div_mse 0.015718732561383928 all_res 0.8067588580765187

#  --style_mix_ratio=0.4 \
#  --style_mix_strength=0.3 \
#  --style_maintain_strength=0.2 all_mse 0.005110447142275641 all_div_mse 0.007429804120744977 all_res 0.6951192577713898