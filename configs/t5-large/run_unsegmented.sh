accelerate launch --config_file=./configs/t5-large/accelerate_config.yaml src/epik_eval/__main__.py \
    --model_name='google/t5-v1_1-large' \
    --model_type='t5' \
    --epochs=5000 \
    --eval_every_x_epoch=50 \
    --lr=1e-4 \
    --warmup_steps=3600 \
    --total_batch_size=50 \
    --minibatch_size=50 \
    --gradient_accumulation_steps=1 \
    --eval_batch_size=900 \
    --model_answers_csv='./logs/t5-large_unsegmented.csv' \
    --answer_max_length=100 \
    --checkpoint_directory='./checkpoints/t5-large_unsegmented' \
    --checkpoint_every=50 \
    --seed=0
