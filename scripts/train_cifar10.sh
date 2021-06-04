for i in 1 2 3 4 5; do
    python Benchmarks/train_cifar.py --config 'configs/CIFAR10/default.json' --name ${i} --seed ${i}
done
