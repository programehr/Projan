echo '' > gridsearch2.txt
for i in {1..8}
do
for j in {1..8}
do
#######
#mnist
#######
w2=$(python -c "print ($i/4)")
w3=$(python -c "print ($j/4)")
 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --verbose 1 --dataset mnist --model net --attack prob --device cuda --epoch 100 --save \
 --validate_interval 100 \
 --losses loss1 loss2_8 loss3_9 \
 --init_loss_weights 1.0 $w2 $w3 \
 --probs 0.33 0.33 0.34 \
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
 >> gridsearch2.txt  
done
done
