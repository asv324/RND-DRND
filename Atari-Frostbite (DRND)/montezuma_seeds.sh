#!/bin/bash

# Loop through seed values 0 to 4
for seed in {0..4}
do
    echo "Running training with seed $seed..."
    docker run --rm -d -v "C:\Users\asvet\Important Stuff\DRND\online":/app --gpus device=0 drnd python /app/train_montezuma.py --seed $seed &
done

wait
echo "Training complete."