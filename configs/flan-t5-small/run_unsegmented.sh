accelerate launch --config_file=./configs/flan-t5-small/accelerate_config.yaml src/run.py \
    --model_name='google/flan-t5-small' \
    --model_type='t5' \
    --epochs=5000 \
    --eval_every_x_epoch=50 \
    --lr=1e-4 \
    --warmup_steps=3600 \
    --total_batch_size=50 \
    --minibatch_size=50 \
    --gradient_accumulation_steps=1 \
    --eval_batch_size=3600 \
    --model_answers_csv='./logs/flan-t5-small_unsegmented.csv' \
    --answer_max_length=100 \
    --checkpoint_directory='./checkpoints/flan-t5-small_unsegmented' \
    --checkpoint_every=50 \
    --seed=0
