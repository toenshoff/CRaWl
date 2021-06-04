for i in 0 1 2 3 4; do
    python Benchmarks/train_zinc.py --config 'configs/ZINC/default.json' --name ${i} --seed ${i}
done
