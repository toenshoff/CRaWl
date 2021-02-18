for i in 1 2 3 4 5; do
    python OGB/train_ogb.py --data MOLHIV --config configs/MOLHIV/vn.json --name ${i} --seed ${i}
done