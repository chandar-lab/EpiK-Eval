<div style="text-align: center;">
<h1>EpiK-Eval: Evaluation for Language Models as Epistemic Models</h1>
<img src="EpiK-Eval_logo.webp" alt="EpiK-Eval Logo" width="200"/>

[Paper](https://arxiv.org/abs/2310.15372) | [Blog Post](https://gabprato.github.io/epik-eval/)
</div>

Benchmark to evaluate the capability of language models to recall and consolidate information from multiple training documents in order to answer questions during inference.

## Requirements
- python >= 3.10.0
- accelerate == 0.24.1
- deepspeed >= 0.12.3
- protobuf >= 4.24.4
- pytorch >= 2.1.0
- sentencepiece >= 0.1.99
- transformers >= 4.35.1
- wandb >= 0.15.12 *(Optional)*

## Install
⚠️ **Warning:** Installing this package will patch the `accelerate` package. To avoid conflicts, please use a dedicated virtual environment if `accelerate` is used in other projects, and activate it before installing this repo.
```bash
git clone https://github.com/chandar-lab/EpiK-Eval.git
cd EpiK-Eval
make
pip install --no-deps .
```

## Run the benchmark
To benchmark a model on unsegmented stories:
```bash
bash configs/<model>/run_unsegmented.sh
```
or on segmented stories:
```bash
bash configs/<model>/run_segmented.sh
```

## Dataset Privacy Notice
Our dataset is provided in an encrypted format and is decrypted only during the installation process. To avoid this data leaking into training corpuses, we urge all users to exercise caution and ensure they do not inadvertently publish or push the unencrypted dataset online. As a precautionary measure, the unencrypted dataset is included in the `.gitignore` file. Your cooperation in maintaining the privacy of our dataset is greatly appreciated.