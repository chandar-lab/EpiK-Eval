accelerate launch --config_file=./configs/flan-t5-small/accelerate_config.yaml src/run.py \
    --segment_documents \
    --model_name='google/flan-t5-small' \
    --model_type='t5' \
    --epochs=2093 \
    --eval_every_x_epoch=21 \
    --lr=1e-4 \
    --warmup_steps=3600 \
    --total_batch_size=50 \
    --minibatch_size=50 \
    --gradient_accumulation_steps=1 \
    --eval_batch_size=2150 \
    --model_answers_csv='./logs/flan-t5-small_segmented.csv' \
    --answer_max_length=100 \
    --checkpoint_directory='./checkpoints/flan-t5-small_segmented' \
    --checkpoint_every=21 \
    --seed=0
