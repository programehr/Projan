#echo '' > attack_prob_cifar_multirun2.txt
#echo '' > defense_di_attack_prob_cifar_multirun2.txt
#echo '' > defense_nc_attack_prob_cifar_multirun2.txt
#echo '' > attack_badnet_cifar_multirun2.txt
#echo '' > defense_di_attack_badnet_cifar_multirun2.txt
#echo '' > defense_nc_attack_badnet_cifar_multirun2.txt

#echo '' > attack_prob_mnist_multirun2.txt 
#echo '' > defense_di_attack_prob_mnist_multirun2.txt
#echo '' > defense_nc_attack_prob_mnist_multirun2.txt
#echo '' > attack_badnet_mnist_multirun2.txt 
#echo '' > defense_di_attack_badnet_mnist_multirun2.txt
#echo '' > defense_nc_attack_badnet_mnist_multirun2.txt
#######
#cifar10
#######
for i in {0..9}
do
 CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_attack.py --verbose 1 --dataset cifar10 --model resnet18_comp --attack prob --device cuda --epoch 200 \
 --lr 0.001 --save\
 --losses loss1 loss2_8 loss3_9 \
 --init_loss_weights 0.7 1.0 0.5 \
 --poison_percent 0.1 \
 --pretrain \
 --cbeta_epoch 0 \
 --mark_path square_white.png \
 --mark_height 3 \
 --mark_width 3 \
 --height_offset 2 \
 --width_offset 2 \
 --batch_size 100 \
 --extra_mark "mark_path=square_white.png mark_height=3 mark_width=3 height_offset=10 width_offset=10" \
 >> attack_prob_cifar_multirun2.txt
 
 CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack prob --defense neuron_inspect --random_init --device cuda --save >> defense_ni_attack_prob_cifar_multirun2.txt

 CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack prob --defense tabor --random_init --device cuda --save >> defense_tabor_attack_prob_cifar_multirun2.txt

CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_attack.py --verbose 1 --dataset cifar10 --model resnet18_comp --attack badnet --device cuda --epoch 200 --save --mark_path square_white.png  --mark_height 3  --mark_width 3  --height_offset 2  --width_offset 2  --batch_size 100 --pretrain >> attack_badnet_cifar_multirun2.txt

 CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack badnet --defense neuron_inspect --random_init --device cuda --save >> defense_ni_attack_badnet_cifar_multirun2.txt

 CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack badnet --defense tabor --random_init --device cuda --save >> defense_tabor_attack_badnet_cifar_multirun2.txt
#######
#mnist
#######
 CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_attack.py --verbose 1 --dataset mnist --model net --attack prob --device cuda --epoch 200 --save \
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
 >> attack_prob_mnist_multirun2.txt 
 
  CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack prob --defense neuron_inspect --random_init --device cuda --save >> defense_ni_attack_prob_mnist_multirun2.txt

 CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack prob --defense tabor --random_init --device cuda --save >> defense_tabor_attack_prob_mnist_multirun2.txt
 
CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_attack.py --verbose 1 --dataset mnist --model net --attack badnet --device cuda --epoch 200 --save --mark_path square_white.png  --mark_height 3  --mark_width 3  --height_offset 2  --width_offset 2  --batch_size 100 >> attack_badnet_mnist_multirun2.txt 

  CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack badnet --defense neuron_inspect --random_init --device cuda --save >> defense_ni_attack_badnet_mnist_multirun2.txt

 CUDA_VISIBLE_DEVICES=1 python3.9 ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack badnet --defense tabor --random_init --device cuda --save >> defense_tabor_attack_badnet_mnist_multirun2.txt
 
 echo $i
done
