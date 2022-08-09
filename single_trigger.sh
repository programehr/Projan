for i in {0..19}
do 
 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --verbose 1 --dataset mnist --model net --attack prob --device cuda --epoch 5 --save \
 --losses loss1 loss2_8 loss3_9 \
 --init_loss_weights 1 1 0 \
 --probs 0.7 \
 --validate_interval 5 \
 --poison_percent 0.1 \
 --cbeta_epoch 0 \
 --mark_path square_white.png \
 --mark_height 3 \
 --mark_width 3 \
 --height_offset 2 \
 --width_offset 2 \
 --batch_size 100 \
 --lr 1e-2 \
 >> single-mark.txt
 
CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack prob --defense neural_cleanse --random_init --device cuda --save >> defense_nc_single.txt

done
