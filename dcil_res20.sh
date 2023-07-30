declare datapath=~/Data/CIFAR

for d in cifar10 cifar100; do
  for p in 0.9 0.95; do
    for s in 6 7 8; do
      python run_dcil.py ${d} --datapath ${datapath} -a resnet --layers 20 -C -g 0 -P --prune-type unstructured \
      --prune-freq 16 --prune-rate ${p} --prune-imp L2 --epochs 300 --batch-size 128  --lr 0.2 --warmup-lr-epoch 5 \
      --wd 1e-4 --nesterov --scheduler multistep --milestones 150 225  --gamma 0.1 --cu_num 0 --warmup-loss 70 \
      --target_epoch 225 --save dcil_sparsity${p}_seed${s}.pth  | tee log/dcil_res20_${d}_sparsity${p}_seed${s}.txt
    done
  done
done