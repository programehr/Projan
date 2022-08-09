echo '' > attack_prob_cifar_multirun2.txt
echo '' > defense_di_attack_prob_cifar_multirun2.txt
echo '' > defense_nc_attack_prob_cifar_multirun2.txt
echo '' > defense_abs_attack_prob_cifar_multirun2.txt

echo '' > attack_badnet_cifar_multirun2.txt
echo '' > defense_di_attack_badnet_cifar_multirun2.txt
echo '' > defense_nc_attack_badnet_cifar_multirun2.txt
echo '' > defense_abs_attack_badnet_cifar_multirun2.txt

echo '' > attack_prob_mnist_multirun2.txt 
echo '' > defense_di_attack_prob_mnist_multirun2.txt
echo '' > defense_nc_attack_prob_mnist_multirun2.txt
echo '' > defense_abs_attack_prob_mnist_multirun2.txt

echo '' > attack_badnet_mnist_multirun2.txt 
echo '' > defense_di_attack_badnet_mnist_multirun2.txt
echo '' > defense_nc_attack_badnet_mnist_multirun2.txt
echo '' > defense_abs_attack_badnet_mnist_multirun2.txt
#######
#cifar10
#######
for i in {0..9}
do
 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --verbose 1 --dataset cifar10 --model resnet18_comp --attack prob --device cuda --epoch 200 \
 --lr 0.001 --save\
 --losses loss1 loss2_8 loss3_9 \
 --init_loss_weights 1 0.7 0.3 \
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
 
 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack prob --defense deep_inspect --random_init --device cuda --save >> defense_di_attack_prob_cifar_multirun2.txt

 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack prob --defense neural_cleanse --nc-epoch 50 --random_init --device cuda --save >> defense_nc_attack_prob_cifar_multirun2.txt
 
CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack prob --defense abs --random_init --device cuda --save >> defense_abs_attack_prob_cifar_multirun2.txt

CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --verbose 1 --dataset cifar10 --model resnet18_comp --attack badnet --device cuda --epoch 200 --save --mark_path square_white.png  --mark_height 3  --mark_width 3  --height_offset 2  --width_offset 2  --batch_size 100 --pretrain >> attack_badnet_cifar_multirun2.txt

 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack badnet --defense deep_inspect --random_init --device cuda --save >> defense_di_attack_badnet_cifar_multirun2.txt

 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack badnet --defense neural_cleanse  --nc-epoch 50 --random_init --device cuda --save >> defense_nc_attack_badnet_cifar_multirun2.txt
 
 
CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack badnet --defense abs --random_init --device cuda --save >> defense_abs_attack_badnet_cifar_multirun2.txt
#######
#mnist
#######
 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --verbose 1 --dataset mnist --model net --attack prob --device cuda --epoch 200 --save \
 --losses loss1 loss2_8 loss3_9 \
 --init_loss_weights 1 0.7 0.3 \
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
 
  CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack prob --defense deep_inspect --random_init --device cuda --save >> defense_di_attack_prob_mnist_multirun2.txt

 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack prob --defense neural_cleanse  --nc-epoch 50 --random_init --device cuda --save >> defense_nc_attack_prob_mnist_multirun2.txt
 
 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack prob --defense abs --random_init --device cuda --save >> defense_abs_attack_prob_mnist_multirun2.txt
 
CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_attack.py --verbose 1 --dataset mnist --model net --attack badnet --device cuda --epoch 200 --save --mark_path square_white.png  --mark_height 3  --mark_width 3  --height_offset 2  --width_offset 2  --batch_size 100 >> attack_badnet_mnist_multirun2.txt 

  CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack badnet --defense deep_inspect --random_init --device cuda --save >> defense_di_attack_badnet_mnist_multirun2.txt

 CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack badnet --defense neural_cleanse  --nc-epoch 50 --random_init --device cuda --save >> defense_nc_attack_badnet_mnist_multirun2.txt
 
  CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --verbose 1 --validate_interval 1 --dataset mnist --model net --attack badnet --defense abs --random_init --device cuda --save >> defense_abs_attack_badnet_mnist_multirun2.txt
 
done
