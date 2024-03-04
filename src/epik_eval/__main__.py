import argparse
import json
import os
import pathlib
import shutil
import time
import torch
try:
    import wandb
    from accelerate.tracking import WandBTracker
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False
    print("Warning: wandb isn't installed. Script can still run without passing --wandb as an argument.")
from accelerate import Accelerator
from accelerate.utils import set_seed, GradientAccumulationPlugin
from dataloader import TrainDataLoader, EvaluationDataLoader
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, T5Tokenizer, T5ForConditionalGeneration
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def _run(accelerator, start_time, args):
    device = accelerator.device

    assert args.model_type in ['causal', 't5'], f"Expected model_type to be either 'causal' or 't5', but got: {args.model_type}"
    if args.model_type == 'causal':
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        train_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        eval_tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left') # For correct generation, padding must be left
        if train_tokenizer.pad_token_id is None:
            train_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            eval_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(train_tokenizer))
    elif args.model_type == 't5':
        # Although T5 can handle sequences of any length, it has been
        # pretrained only on sequences of max 512 tokens, so it might not
        # perform well on inputs of length greater than 512 unless fine-tuned.
        model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
        train_tokenizer = T5Tokenizer.from_pretrained(
            args.model_name,
            legacy=False,
            model_max_length=1e10)  # default max length is 512
        eval_tokenizer = train_tokenizer
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(
            args.adam_beta1,
            args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss(ignore_index=train_tokenizer.pad_token_id)
    
    train_dataloader = TrainDataLoader(
        train_tokenizer,
        args.model_type,
        args.segment_documents,
        args.total_batch_size)
    
    eval_dataloader = EvaluationDataLoader(
        eval_tokenizer,
        args.segment_documents,
        args.eval_batch_size,
        args.model_type)
    
    if args.lr_schedule == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=int(args.epochs * len(train_dataloader.dataset) / args.total_batch_size))
    elif args.lr_schedule == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=int(args.epochs * len(train_dataloader.dataset) / args.total_batch_size))
    else:
        raise ValueError(f"Expected lr_schedule to be either 'constant', 'linear' or 'cosine', but got: '{args.lr_schedule}'.")

    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader)

    last_checkpoint_dir = os.path.join(args.checkpoint_directory, 'last_checkpoint/')
    if os.path.isdir(last_checkpoint_dir):
        accelerator.print(f"Last checkpoint found. Resuming from: {last_checkpoint_dir}")
        metrics = _load_checkpoint(last_checkpoint_dir, accelerator)
        best_val_accuracy = metrics['best_val_accuracy']
        selected_train_accuracy = metrics['selected_train_accuracy']
        selected_test_accuracy = metrics['selected_test_accuracy']
        selected_test_acc_per_task = metrics['selected_test_acc_per_task']
        best_epoch = metrics['best_epoch']
        starting_epoch = metrics['current_epoch'] + 1
        elapsed_time = metrics['elapsed_time']
    else:
        accelerator.print(f"Found no checkpoint to resume from. Starting from scratch.")
        if args.eval_at_start:
            _ = _evaluate(accelerator, model, eval_dataloader, eval_tokenizer, device, args, 0)
        best_val_accuracy = -1
        selected_train_accuracy = -1
        selected_test_accuracy = -1
        selected_test_acc_per_task = {}
        best_epoch = -1
        starting_epoch = 1
        elapsed_time = 0

    tqdm_bar = tqdm(range(starting_epoch, args.epochs + 1), desc='Epoch', disable=(args.no_tqdm or not accelerator.is_main_process))
    for epoch in tqdm_bar:
        train_loss = _train_epoch(accelerator, model, train_dataloader, optimizer, lr_scheduler, criterion, device, args)

        if (epoch % args.eval_every_x_epoch) == 0:
            accuracies = _evaluate(accelerator, model, eval_dataloader, eval_tokenizer, device, args, epoch)
            if accelerator.is_main_process:
                if accuracies['val'] > best_val_accuracy:
                    best_val_accuracy = accuracies['val']
                    selected_train_accuracy = accuracies['train']
                    selected_test_accuracy = accuracies['test']
                    selected_test_acc_per_task = accuracies['test_per_task']
                    best_epoch = epoch
                tqdm_bar.set_postfix({'train loss': train_loss,
                                      'train accuracy': accuracies['train'],
                                      'val accuracy': accuracies['val']})
                if args.wandb:
                    accelerator.log({'Train Loss': train_loss,
                                     'Train Accuracy (Memorization)': accuracies['train'],
                                     'Val Accuracy': accuracies['val']} | accuracies['val_per_task'],
                                    step=epoch)
        elif accelerator.is_main_process:
            tqdm_bar.set_postfix({'train loss': train_loss})
            if args.wandb:
                accelerator.log({'Train Loss': train_loss}, step=epoch)

        if (epoch % args.checkpoint_every) == 0:
            _save_checkpoint(
                args.checkpoint_directory,
                accelerator,
                best_val_accuracy,
                selected_train_accuracy,
                selected_test_accuracy,
                selected_test_acc_per_task,
                best_epoch,
                epoch,
                elapsed_time +
                time.time() -
                start_time)

    if accelerator.is_main_process and args.wandb:
        accelerator.log({'Test Accuracy': selected_test_accuracy} | selected_test_acc_per_task, step=epoch)
        accelerator.end_training()  # Tells trackers such as wandb to end
    accelerator.print(
        f'Benchmark finished.\n'
        f'Elapsed time: {(elapsed_time + time.time() - start_time)/60:.2f} minutes\n'
        f'Test Accuracy: {selected_test_accuracy:.2f}% (Epoch {best_epoch}, Train Accuracy {selected_train_accuracy:.2f}%, Val Accuracy {best_val_accuracy:.2f}%)\n'
        f'Model answers recorded in: {args.model_answers_csv.replace(".csv", "_epoch%.csv")}')


def _train_epoch(
        accelerator,
        model,
        train_dataloader,
        optimizer,
        lr_scheduler,
        criterion,
        device,
        args):

    model.train()

    # Epoch metrics
    epoch_loss = 0
    epoch_samples = 0

    for batch in iter(train_dataloader):
        if args.model_type == 'causal':
            batch, attention_mask, seq_length, _ = batch
        elif args.model_type == 't5':
            batch, attention_mask, seq_length, target, target_length, _ = batch

        # Iterate minibatches. Minibatches are per device, hence we iterate here.
        for i in range(0, batch.shape[0], args.minibatch_size):
            with accelerator.accumulate(model):
                # minimize padding
                minibatch_max_len = seq_length[i: i + args.minibatch_size].max().item()
                minibatch = batch[i: i + args.minibatch_size, :minibatch_max_len]  # minimizes padding
                mini_attention_mask = attention_mask[i: i + args.minibatch_size, :minibatch_max_len]

                if args.model_type == 'causal':
                    output = model(input_ids=minibatch, attention_mask=mini_attention_mask)

                    # causal language model must predict next word
                    logits, mini_target = output.logits[:, :-1], minibatch[:, 1:]
                    # (N, C), (N)
                    logits, mini_target = logits.reshape(-1, logits.shape[-1]), mini_target.reshape(-1)
                    loss = criterion(logits, mini_target)
                elif args.model_type == 't5':
                    mini_target_max_len = target_length[i: i + args.minibatch_size].max().item()
                    mini_target = target[i: i + args.minibatch_size, :mini_target_max_len].contiguous()

                    loss = model(input_ids=minibatch, attention_mask=mini_attention_mask, labels=mini_target).loss

                epoch_loss += loss.item() * minibatch.shape[0]  # Undo the mean
                epoch_samples += minibatch.shape[0]
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.model_type == 'causal':
                    del loss, logits, output  # Free up memory
                elif args.model_type == 't5':
                    del loss

    return epoch_loss / epoch_samples  # mean epoch loss


def _evaluate(
        accelerator,
        model,
        eval_dataloader,
        eval_tokenizer,
        device,
        args,
        epoch):
    model.eval()

    # Evaluation
    results = []
    for batch in iter(eval_dataloader):
        prompt_id, prompt, attention_mask, prompt_len, nonduplicate_mask = batch

        # minimize padding
        max_len = prompt_len.max().item()
        prompt = prompt[:, -max_len:]  # padding is left side, so keep max len on right side
        attention_mask = attention_mask[:, -max_len:]
        
        with torch.no_grad():
            output = model.generate(
                input_ids=prompt,
                attention_mask=attention_mask,
                max_new_tokens=args.answer_max_length,
                pad_token_id=eval_tokenizer.pad_token_id,
                synced_gpus=args.synced_gpus)
        
        results.append((prompt_id, output, nonduplicate_mask))

    results = accelerator.pad_across_processes(
        results,
        dim=1,
        pad_index=eval_tokenizer.pad_token_id,
        pad_first=eval_tokenizer.padding_side == 'left')
    results = accelerator.gather(results)

    # Logging    
    if accelerator.is_main_process:
        correct = [0] * len(eval_dataloader.dataset.targets)
        model_answers = [None] * len(eval_dataloader.dataset.targets)
        for prompt_id, output, nonduplicate_mask in results:
            prompt_id, output = prompt_id[nonduplicate_mask], output[nonduplicate_mask]
            non_padding = output != eval_tokenizer.pad_token_id
            for i in range(output.shape[0]):
                # Remove padding, bos_token if present and eos_token (always present)
                model_answer = eval_tokenizer.decode(output[i, non_padding[i]][1 if eval_tokenizer.bos_token_id is not None else 0: -1])
                
                if args.model_type == 'causal':
                    # remove the prompt part and the extra character between the prompt and the answer
                    prompt_len = len(eval_dataloader.dataset.prompts[prompt_id[i]]) + 1
                    model_answers[prompt_id[i]] = model_answer[prompt_len:]
                elif args.model_type == 't5':
                    model_answers[prompt_id[i]] = model_answer

                target_answer = eval_dataloader.dataset.targets[prompt_id[i]]
                if model_answers[prompt_id[i]][:len(target_answer)] == target_answer:
                    correct[prompt_id[i]] = 1
                else:
                    correct[prompt_id[i]] = 0

        # Log model answers
        model_answers_csv = eval_dataloader.dataset.eval_df.copy()
        model_answers_csv.rename(columns={'answer': 'target answer'}, inplace=True)
        model_answers_csv['model answer'] = model_answers
        model_answers_csv['correct'] = correct

        accuracies = {}
        for set in ['train', 'val', 'test']:
            questions = model_answers_csv.query("`set` == @set")
            accuracies[set] = 100 * questions['correct'].sum() / len(questions['correct'])
            acc_per_task = {}
            for task in questions['task'].unique():
                task_questions = questions[questions['task'] == task]
                acc_per_task[f'{set.capitalize()} Task {task} Accuracy'] = 100 * task_questions['correct'].sum() / len(task_questions['correct'])
            accuracies[f'{set}_per_task'] = acc_per_task

        csv_path = pathlib.Path(args.model_answers_csv)
        csv_path = csv_path.with_stem(f"{csv_path.stem}_epoch{epoch}")
        pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        model_answers_csv.to_csv(csv_path, sep='|')

        return accuracies
    return None


def _load_checkpoint(checkpoint_directory, accelerator):
    accelerator.load_state(checkpoint_directory)
    with open(os.path.join(checkpoint_directory, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
    return metrics


def _save_checkpoint(
        checkpoint_directory,
        accelerator,
        best_val_accuracy,
        selected_train_accuracy,
        selected_test_accuracy,
        selected_test_acc_per_task,
        best_epoch,
        current_epoch,
        elapsed_time):

    os.makedirs(checkpoint_directory, exist_ok=True)

    penultimate_dir = os.path.join(checkpoint_directory, 'penultimate_checkpoint/')
    last_dir = os.path.join(checkpoint_directory, 'last_checkpoint/')
    temp_dir = os.path.join(checkpoint_directory, 'temp_checkpoint/')

    accelerator.save_state(temp_dir)

    if accelerator.is_main_process:
        if os.path.isdir(last_dir):
            if os.path.isdir(penultimate_dir):
                shutil.rmtree(penultimate_dir)
            os.rename(last_dir, penultimate_dir)

        os.rename(temp_dir, last_dir)

        metrics = {
            'best_val_accuracy': best_val_accuracy,
            'selected_train_accuracy': selected_train_accuracy,
            'selected_test_accuracy': selected_test_accuracy,
            'selected_test_acc_per_task': selected_test_acc_per_task,
            'best_epoch': best_epoch,
            'current_epoch': current_epoch,
            'elapsed_time': elapsed_time
        }
        with open(os.path.join(last_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--segment_documents',
        action='store_true',
        help="Will run the benchmark with documents segmented into separate samples.")
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m')
    parser.add_argument(
        '--model_type',
        type=str,
        default='causal',
        help="Currently supported: 'causal' or 't5'.")
    parser.add_argument(
        '--answer_max_length',
        type=int,
        default=150,
        help="The maximum number of tokens the model can generate for his answer.")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--lr_schedule',
        type=str,
        default='constant',
        help="'constant', 'linear' or 'cosine'. Both 'linear' and 'cosine' will decay from lr to 0.")
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=0,
        help='Number of learning rate warmup steps. Will linearly warmup from 0 to the learning rate.')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval_every_x_epoch', type=int, default=1)
    parser.add_argument(
        '--total_batch_size',
        type=int,
        help='Total training batch size. Needs to be divisible by number of ranks. The batch size per rank is total_batch_size / num_ranks.')
    parser.add_argument(
        '--minibatch_size',
        type=int,
        default=1,
        help='Training minibatch size per rank. Minibatch gradients are accumulated until the full batch is processed.')
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        help='This should be the total batch size / number of GPUs / minibatch size. '
        "For example, if the total batch size is 32 and you are using 4 GPUs, each GPU gets 8 samples. If you're minibatch size is set to 2, you are accumulating gradient 4 times.")
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=128,
        help='Value should be a multiple of the number of ranks.')
    parser.add_argument(
        '--model_answers_csv',
        type=str,
        help='CSV where model answers will be recorded.',
        default='./logs/last_run.csv')
    parser.add_argument(
        '--checkpoint_directory',
        type=str,
        help='Directory where checkpoints will be saved. Only two checkpoints are kept, the penultimate and the last one.',
        default='./checkpoints/last_run/')
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        help='Will save a checkpoint every x epoch. Only the last two checkpoints are kept, older checkpoints are deleted.',
        default=10)
    parser.add_argument(
        '--eval_at_start',
        action='store_true',
        help='Will run an evaluation before the benchmark begins.')
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument(
        '--wandb_project',
        type=str,
        help='Weights & Biases Project Name.')
    parser.add_argument(
        '--wandb_run_id',
        type=str,
        help='Unique wandb run ID. Max 64 characters. Needed for wandb to resume logging of existing experiment.')
    parser.add_argument(
        '--seed',
        type=int,
        help='If not None, will set the seed for random, numpy, torch, torch.cuda and if TPUs are available torch_xla’s cuda state.')
    args = parser.parse_args()

    start_time = time.time()

    if args.wandb:
        assert wandb_available, "Script launched with --wandb argument, but wandb module isn't installed."

    accelerator = Accelerator(
        split_batches=True,
        dispatch_batches=True,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_with_dataloader=False),
        log_with='wandb' if args.wandb else None)
    accelerator.print(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    args.synced_gpus = (accelerator.state.deepspeed_plugin is not None and 
                        accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']['stage'] == 3)

    # Print hyperparameters
    for key, value in args.__dict__.items():
        accelerator.print(f'{key}: {value}')
    accelerator.print()
    args.tqdm = not args.no_tqdm

    # Initialize wandb
    if accelerator.is_main_process and args.wandb:
        wandb_kwargs = {
            'id': args.wandb_run_id,
            'resume': 'allow',
            'allow_val_change': True
        }
        accelerator.init_trackers(
            project_name=args.wandb_project,
            init_kwargs={'wandb': wandb_kwargs})
        wandb.config.update(args, allow_val_change=wandb_kwargs['allow_val_change'])

    _run(accelerator, start_time, args)


if __name__ == '__main__':
    main()