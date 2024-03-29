accelerate launch --config_file=./configs/opt-125m/accelerate_config.yaml src/epik_eval/__main__.py \
    --segment_documents \
    --model_name='facebook/opt-125m' \
    --model_type='causal' \
    --epochs=2093 \
    --eval_every_x_epoch=21 \
    --lr=6e-5 \
    --warmup_steps=3600 \
    --total_batch_size=50 \
    --minibatch_size=50 \
    --gradient_accumulation_steps=1 \
    --eval_batch_size=2150 \
    --model_answers_csv='./logs/opt-125m_segmented.csv' \
    --answer_max_length=100 \
    --checkpoint_directory='./checkpoints/opt-125m_segmented' \
    --checkpoint_every=21 \
    --seed=0
