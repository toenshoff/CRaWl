# we recommend to parallelize the training runs on a compute cluster
for i in 0 1 2 3 4 5 6 7 8 9; do
    python OGB/train_ogb.py --data MOLPCBA --config 'configs/MOLPCBA/default.json' --name ${i} --seed ${i}
done
