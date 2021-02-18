for i in 1 2 3 4 5; do
    python Benchmarks/train_zinc.py --config configs/ZINC/vn.json --name ${i} --seed ${i}
done