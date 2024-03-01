<div align="center">
<h1>EpiK-Eval: Evaluation for Language Models as Epistemic Models</h1>
<img src="EpiK-Eval_logo.webp" alt="EpiK-Eval Logo" width="200"/>

[Paper](https://arxiv.org/abs/2310.15372) | [Blog Post](https://gabprato.github.io/epik-eval/)
</div>

Benchmark to evaluate the capability of language models to consolidate and recall information from multiple training documents in order to answer questions during inference.

## Requirements
- python >= 3.10.0
- accelerate == 0.24.1
- deepspeed >= 0.12.3
- pandas >= 2.1.1
- protobuf >= 4.24.4
- pytorch >= 2.1.0
- sentencepiece >= 0.1.99
- transformers >= 4.35.1
- matplotlib *(Optional)*
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

## Manually Decrypt Dataset
Our dataset is automatically decrypted during installation. If for ever reason, one wants to manually decrypt the dataset, for example, without installing our repo, one can run:
```bash
make prepare_dataset
```
For more information on the format of our dataset, see [`data/README.md`](./data/README.md)

## Additional Scripts
We provide the following three additional scripts. Each script is meant to be ran in the `/scripts/` directory, e.g., `python generate_paper_plots.py`.

### Dataset Generation
To generate a new dataset, one can run:
```bash
python generate_dataset.py
```
The default parameters generate a dataset with the same format as the one we provide. Run `python scripts/generate_dataset.py --help` for details.

### Compute Benchmark Metrics
Once a model has been benchmarked, the various metrics shown in our paper can be computed via:
```bash
python compute_paper_metrics.py --model_answer_log=logs/example_log.csv
```

### Generate Paper Plots
To recreate the plots shown in our paper, one can run:
```bash
python generate_paper_plots.py
```
We already provide these figures in `figures/`