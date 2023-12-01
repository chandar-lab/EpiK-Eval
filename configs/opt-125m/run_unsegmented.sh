accelerate launch --config_file=./configs/opt-125m/accelerate_config.yaml src/run.py \
    --model_name='facebook/opt-125m' \
    --model_type='causal' \
    --epochs=5000 \
    --eval_every_x_epoch=50 \
    --lr=6e-5 \
    --warmup_steps=3600 \
    --total_batch_size=50 \
    --minibatch_size=50 \
    --gradient_accumulation_steps=1 \
    --eval_batch_size=3600 \
    --model_answers_csv='./logs/opt-125m_unsegmented.csv' \
    --answer_max_length=100 \
    --checkpoint_directory='./checkpoints/opt-125m_unsegmented' \
    --checkpoint_every=50 \
    --seed=0
