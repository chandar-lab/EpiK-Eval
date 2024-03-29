accelerate launch --config_file=./configs/opt-2.7b/accelerate_config.yaml src/epik_eval/__main__.py \
    --model_name='facebook/opt-2.7b' \
    --model_type='causal' \
    --epochs=5000 \
    --eval_every_x_epoch=50 \
    --lr=1.6e-5 \
    --warmup_steps=3600 \
    --total_batch_size=50 \
    --minibatch_size=25 \
    --gradient_accumulation_steps=2 \
    --eval_batch_size=200 \
    --model_answers_csv='./logs/opt-2.7b_unsegmented.csv' \
    --answer_max_length=100 \
    --checkpoint_directory='./checkpoints/opt-2.7b_unsegmented' \
    --checkpoint_every=50 \
    --seed=0
