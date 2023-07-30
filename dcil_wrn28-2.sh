declare datapath=~/Data/CIFAR

for d in cifar10 cifar100; do
  for p in 0.9 0.95 0.99; do
    for s in 6 7 8; do
      python run_dcil.py ${d} --datapath ${datapath} -a wideresnet --layers 28 --width-mult 2 -C -g 0 -P --prune-type unstructured \
      --prune-freq 16 --prune-rate ${p} --prune-imp L2 --epochs 200 --batch-size 128  --lr 0.1 --warmup-lr-epoch 5 \
      --wd 5e-4 --nesterov --scheduler multistep --milestones 60 120 160 --gamma 0.2 --cu_num 0 --warmup-loss 70 \
      --target_epoch 120 --save dcil_sparsity${p}_seed${s}.pth | tee log/dcil_wrn28-2_${d}_sparsity${p}_seed${s}.txt
    done
  done
done
