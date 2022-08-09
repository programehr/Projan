for i in {0..9}
do
#######
#mnist
#######
 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --verbose 1 --dataset mnist --model net --attack prob --device cuda --epoch 200 --save \
 --losses loss1 loss2_8 loss3_9 \
 --init_loss_weights 0.7 1.0 0.5 \
 --poison_percent 0.1 \
 --cbeta_epoch 0 \
 --mark_path square_white.png \
 --mark_height 3 \
 --mark_width 3 \
 --height_offset 2 \
 --width_offset 2 \
 --batch_size 100 \
 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=20 width_offset=20" \
 >> attack_prob_mnist_multirun4.txt 
 
 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack prob --defense neural_cleanse --random_init --device cuda --save >> defense_nc_attack_prob_mnist_multirun4.txt
 
  CUDA_VISIBLE_DEVICES=0 python ./tests/cleanse.py --verbose 1 --dataset mnist --model net --attack prob --device cuda --save >> cleanse_prob4.txt
 
done
