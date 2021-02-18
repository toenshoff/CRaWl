# we recommend to parallelize the training runs on compute cluster
for i in 1 2 3 4 5; do
    python OGB/train_ogb.py --data MOLPCBA --config configs/MOLPCBA/default.json --name ${i} --seed ${i}
done