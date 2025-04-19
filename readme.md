resnet34:

python train.py --epochs 100 -lr 5e-4 -b 64 --weight_decay 0.0001


swin-t-v2:
python train.py --optimizer adamw_diff_layer --epochs 100 -lr 5e-4 -b 16 --weight_decay 1e-3 --early_stop_patience 8

regnet-y-16gf;
python train.py --epochs 100 -lr 8e-5 -b 8 --weight_decay 5e-4 --grad_clip 0.7    原始

python train.py --epochs 100 -lr 8e-5 -b 8 --weight_decay 1e-3 --grad_clip 0.5 更改過head 有1024的

