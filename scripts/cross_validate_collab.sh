for i in 0 1 2 3 4 5 6 7 8 9; do
    python TUD/train_tud.py --data COLLAB --name ${i} --config 'configs/COLLAB/default.json' --fold ${i} --seed ${i} --split_dir splits/TUD_Test_Splits
done
python TUD/test_tud.py --model_dir 'models/COLLAB/default'