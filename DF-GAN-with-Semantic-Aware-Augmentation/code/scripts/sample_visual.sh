# configs of different datasets
cfg=$1

# model settings
imgs_per_sent=12
cuda=True
gpu_id=0

python src/sample_visual.py \
        --cfg $cfg \
        --imgs_per_sent $imgs_per_sent \
        --cuda $cuda \
        --gpu_id $gpu_id \
