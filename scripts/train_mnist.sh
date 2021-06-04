for i in 0 1 2 3 4; do
    python Benchmarks/train_mnist.py --config 'configs/MNIST/default.json' --name ${i} --seed ${i}
done
