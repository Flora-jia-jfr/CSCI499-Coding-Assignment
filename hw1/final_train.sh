python train.py \
    --in_data_fn=lang_to_sem_data.json \
    --model_output_dir=experiments/lstm \
    --batch_size=1024 \
    --num_epochs=50 \
    --val_every=4