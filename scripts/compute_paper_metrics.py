# This script takes a model's answer log as input and computes the metrics presented in the EpiK-Eval paper: https://arxiv.org/abs/2310.

import argparse
import numpy as np
import pandas as pd


# Number of statements to ignore at the end of an answer, for each task. This is to ignore reasoning parts and the final answer and solely count hallucinations in the statements to recall.
STATEMENTS_TO_IGNORE = {1: 1,
                        2: 1,
                        3: 1,
                        4: 2,
                        5: 2,
                        6: 1,
                        7: 1,
                        8: 2,
                        9: 2,
                        10: 1,
                        11: 1,
                        12: 1,
                        13: 2,
                        14: 1,
                        15: 1,
                        16: 1,
                        17: 1,
                        18: 2}
REASONING_KEYWORDS = {4: ['before', 'after'],
                       5: ['Those are'],
                       8: ['='],
                       9: ['='],
                       13: ['is the'],
                       18: ['=']} # keywords that should be in the reasoning statement for each respective task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_answer_log', type=str, help="Path to the model answer log (CSV file).")
    args = parser.parse_args()
    
    return args


def compute_train_hallucination_rate(train_answers):
    '''
    When prompted with the title of a story (unsegmented setup) or the title of a story part (segmented setup), the model is expected to output, respectively, 
    the story or the story part. We take the model's output and count the number of sentences that aren't in the actual story or story part (again respectively) 
    over the total number of sentences in the model's output. This can be interpreted as the probability of a sentence being incorrect, 
    the hallucination rate at the sentence level.
    '''

    hallucinations = 0
    total_sentences = 0
    hallucinations_per_task = {}
    total_sentences_per_task = {}

    for task, target_answer, model_answer in zip(train_answers['task'], train_answers['target answer'], train_answers['model answer']):
        if task not in hallucinations_per_task:
            hallucinations_per_task[task] = 0
            total_sentences_per_task[task] = 0
        
        if '\n' in target_answer: # some tokenizers replace \n with the space character and the sentences in the training samples are separated by '\n'
            target_answer = target_answer.split('.\n')
            model_answer = model_answer.split('.\n')
        else:
            target_answer = target_answer.split('. ')
            model_answer = model_answer.split('. ')
            
        # Measure hallucination
        for sentence in model_answer:
            if sentence not in target_answer:
                hallucinations += 1
                hallucinations_per_task[task] += 1
        total_sentences += len(model_answer)
        total_sentences_per_task[task] += len(model_answer)
            
    print(f'Train hallucination rate: {100 * hallucinations/total_sentences:.2f}%')
    for task in sorted(total_sentences_per_task.keys()):
        print(f'[Task {task}] Train hallucination rate: {100 * hallucinations_per_task[task]/total_sentences_per_task[task]:.2f}%')


def compute_test_accuracy(test_answers):
    '''
    We report the following metrics with respect to the test q/a:
    - Whole Answer Accuracy: Percentage of model answers that match the target.
    - Recall Accuracy: Percentage of recall parts in the model answer that match the target.
    - Reasoning Accuracy: Percentage of reasoning parts in the model answer that match the target, only considering the set of answers where the recall was correct.
    - Final Answer Accuracy: Percentage of final answers that match the target, only considering the set of answers where the recall and (if there is a reasoning part in the task) reasoning was correct.
    '''

    correct_whole = 0
    correct_recall = 0
    correct_reasoning = 0
    correct_final_answer = 0
    total_reasoning = 0
    total_final_answer = 0
    correct_whole_per_task = {}
    correct_recall_per_task = {}
    correct_reasoning_per_task = {}
    correct_final_answer_per_task = {}
    total_answers_per_task = {}
    total_reasoning_per_task = {}
    total_final_answer_per_task = {}
    for task in STATEMENTS_TO_IGNORE.keys():
        correct_whole_per_task[task] = 0
        correct_recall_per_task[task] = 0
        correct_reasoning_per_task[task] = 0
        correct_final_answer_per_task[task] = 0
        total_answers_per_task[task] = 0
        total_reasoning_per_task[task] = 0
        total_final_answer_per_task[task] = 0

    for task, target_answer, model_answer in zip(test_answers['task'], test_answers['target answer'], test_answers['model answer']):
        total_answers_per_task[task] += 1
        if model_answer == target_answer:
            correct_whole += 1
            correct_whole_per_task[task] += 1

        target_answer = target_answer.split('. ')
        target_recall_part = '. '.join(target_answer[:-STATEMENTS_TO_IGNORE[task]])
        target_reasoning_part = '. '.join(target_answer[-STATEMENTS_TO_IGNORE[task]:-1]) if STATEMENTS_TO_IGNORE[task] > 1 else None
        target_final_part = target_answer[-1]
        
        model_answer = model_answer.split('. ')
        if STATEMENTS_TO_IGNORE[task] == 1: # no reasoning part, only final answer to remove if present
            model_reasoning_part = None
            if 'The answer is' in model_answer[-1]:
                model_recall_part = '. '.join(model_answer[:-1])
                model_final_part = model_answer[-1]
            else:
                model_recall_part = '. '.join(model_answer)
                model_final_part = None
        elif STATEMENTS_TO_IGNORE[task] > 1: # reasoning and final part need to be remove if present
            answer_has_reasoning_part = False
            answer_has_final_part = False
            for keyword in REASONING_KEYWORDS[task]:
                for sentence in model_answer[-STATEMENTS_TO_IGNORE[task]:]:
                    if keyword in sentence:
                        answer_has_reasoning_part = True
                    if 'The answer is' in sentence:
                        answer_has_final_part = True
            
            statements_to_ignore = 0
            if answer_has_reasoning_part:
                statements_to_ignore += STATEMENTS_TO_IGNORE[task] - 1
            if answer_has_final_part:
                statements_to_ignore += 1
            model_recall_part = '. '.join(model_answer[:-statements_to_ignore])
            
            if answer_has_reasoning_part:
                if answer_has_final_part:
                    model_reasoning_part = '. '.join(model_answer[-STATEMENTS_TO_IGNORE[task]:-1])
                else:
                    model_reasoning_part = '. '.join(model_answer[(-STATEMENTS_TO_IGNORE[task]) + 1:])
            else:
                model_reasoning_part = None
                
            if answer_has_final_part:
                model_final_part = model_answer[-1]
            else:
                model_final_part = None
        
        if model_recall_part == target_recall_part:
            correct_recall += 1
            correct_recall_per_task[task] += 1
            
            # When recall is correct, check reasoning if there is one
            if target_reasoning_part is not None:
                total_reasoning += 1
                total_reasoning_per_task[task] += 1
                if model_reasoning_part == target_reasoning_part:
                    correct_reasoning += 1
                    correct_reasoning_per_task[task] += 1

                    # When reasoning is correct, check final answer
                    total_final_answer += 1
                    total_final_answer_per_task[task] += 1
                    if model_final_part == target_final_part:
                        correct_final_answer += 1
                        correct_final_answer_per_task[task] += 1
            else:
                # When recall is correct and there is no reasoning part, check final answer
                total_final_answer += 1
                total_final_answer_per_task[task] += 1
                if model_final_part == target_final_part:
                    correct_final_answer += 1
                    correct_final_answer_per_task[task] += 1

    print(f'Whole Answer Accuracy: {100 * correct_whole/len(test_answers):.2f}%, '\
        f'Recall Accuracy: {100 * correct_recall/len(test_answers):.2f}%, '\
        f'Reasoning Accuracy: {100 * correct_reasoning/total_reasoning:.2f}%, '\
        f'Final Answer Accuracy: {100 * correct_final_answer/total_final_answer:.2f}%')
    for task in sorted(correct_whole_per_task.keys()):
        output = f'[Task {task}] Whole Answer Accuracy: {100 * correct_whole_per_task[task]/total_answers_per_task[task]:.2f}%, ' \
            f'Recall Accuracy: {100 * correct_recall_per_task[task]/total_answers_per_task[task]:.2f}%, '
        
        if STATEMENTS_TO_IGNORE[task] > 1: # means there is a reasoning part to the answer
            if total_reasoning_per_task[task] > 0:
                output += f'Reasoning Accuracy: {100 * correct_reasoning_per_task[task]/total_reasoning_per_task[task]:.2f}%, '
            else:
                output += f'Reasoning Accuracy: 0.00%, '
                
        if total_final_answer_per_task[task] == 0: # All recalls failed case, we report 0 accuracy for the final answer to avoid dividing by zero
            output += f'Final Answer Accuracy: 0.00%'
        else:
            output += f'Final Answer Accuracy: {100 * correct_final_answer_per_task[task]/total_final_answer_per_task[task]:.2f}%'
                
        print(output)


def compute_test_hallucination_rate(test_answers):
    # The hallucination rate for the test q/a is the percentage of sentences in the recall part of the model answer that aren't in the target.
    hallucinations = 0
    total_sentences = 0
    hallucinations_per_task = {}
    total_sentences_per_task = {}
    for task in STATEMENTS_TO_IGNORE.keys():
        hallucinations_per_task[task] = 0
        total_sentences_per_task[task] = 0

    for task, target_answer, model_answer in zip(test_answers['task'], test_answers['target answer'], test_answers['model answer']):    
        target_recall_part = target_answer.split('. ')[:-STATEMENTS_TO_IGNORE[task]]
        model_answer = model_answer.split('. ')
        if STATEMENTS_TO_IGNORE[task] == 1: # no reasoning part, only final answer to remove if present
            if 'The answer is' in model_answer[-1]:
                model_recall_part = model_answer[:-1]
            else:
                model_recall_part = model_answer
        elif STATEMENTS_TO_IGNORE[task] > 1: # reasoning and final part need to be remove if present
            answer_has_reasoning_part = False
            answer_has_final_part = False
            for keyword in REASONING_KEYWORDS[task]:
                for sentence in model_answer[-STATEMENTS_TO_IGNORE[task]:]:
                    if keyword in sentence:
                        answer_has_reasoning_part = True
                    if 'The answer is' in sentence:
                        answer_has_final_part = True
            
            statements_to_ignore = 0
            if answer_has_reasoning_part:
                statements_to_ignore += STATEMENTS_TO_IGNORE[task] - 1
            if answer_has_final_part:
                statements_to_ignore += 1
            model_recall_part = model_answer[:-statements_to_ignore]
        
        # Measure hallucination
        for sentence in model_recall_part:
            if sentence not in target_recall_part:
                hallucinations += 1
                hallucinations_per_task[task] += 1
        total_sentences += len(model_recall_part)
        total_sentences_per_task[task] += len(model_recall_part)
            
    print(f'Test hallucination rate: {100 * hallucinations/total_sentences:.2f}%')
    for task in sorted(total_sentences_per_task.keys()):
        print(f'[Task {task}] Test hallucination rate: {100 * hallucinations_per_task[task]/total_sentences_per_task[task]:.2f}%')


def compute_test_answer_length_distribution(test_answers):
    # Comparison of answer length distribution (recall part only) between model and target, where length is in number of sentences.

    target_len = []
    model_len = []
    target_len_per_task = {}
    model_len_per_task = {}
    for task in STATEMENTS_TO_IGNORE.keys():
        target_len_per_task[task] = []
        model_len_per_task[task] = []

    for task, target_answer, model_answer in zip(test_answers['task'], test_answers['target answer'], test_answers['model answer']):
        target_answer = target_answer.split('. ')[: - STATEMENTS_TO_IGNORE[task]]
        model_answer = model_answer.split('. ')
        if STATEMENTS_TO_IGNORE[task] == 1: # no reasoning part, only final answer to remove if present
            if 'The answer is' in model_answer[-1]:
                model_answer = model_answer[:-1]
        elif STATEMENTS_TO_IGNORE[task] > 1: # reasoning and final part need to be remove if present
            answer_has_reasoning_part = False
            answer_has_final_part = False
            for keyword in REASONING_KEYWORDS[task]:
                for sentence in model_answer[-STATEMENTS_TO_IGNORE[task]:]:
                    if keyword in sentence:
                        answer_has_reasoning_part = True
                    if 'The answer is' in sentence:
                        answer_has_final_part = True
            
            statements_to_ignore = 0
            if answer_has_reasoning_part:
                statements_to_ignore += STATEMENTS_TO_IGNORE[task] - 1
            if answer_has_final_part:
                statements_to_ignore += 1
            if statements_to_ignore > 0:
                model_answer = model_answer[:-statements_to_ignore]
            
        target_len.append(len(target_answer))
        model_len.append(len(model_answer))
        target_len_per_task[task].append(len(target_answer))
        model_len_per_task[task].append(len(model_answer))
        
    target_lengths, target_counts = np.unique(np.asarray(target_len), return_counts=True)
    output = ['Target Count per Length']
    for _len, count in zip(target_lengths, target_counts):
        output.append(f'{_len}: {count}')
    print(', '.join(output))
        
    model_lengths, model_counts = np.unique(np.asarray(model_len), return_counts=True)
    output = ['Model Count per Length']
    for _len, count in zip(model_lengths, model_counts):
        output.append(f'{_len}: {count}')
    print(', '.join(output))

    for task in sorted(target_len_per_task.keys()):
        target_lengths, target_counts = np.unique(np.asarray(target_len_per_task[task]), return_counts=True)
        output = [f'[Task {task}] Target Count per Length']
        for _len, count in zip(target_lengths, target_counts):
            output.append(f'{_len}: {count}')
        print(', '.join(output))

        model_lengths, model_counts = np.unique(np.asarray(model_len_per_task[task]), return_counts=True)
        output = [f'[Task {task}] Model Count per Length']
        for _len, count in zip(model_lengths, model_counts):
            output.append(f'{_len}: {count}')
        print(', '.join(output))


def main():
    args = parse_args()

    answer_csv = pd.read_csv(args.model_answer_logs, sep='|')
    train_answers = answer_csv[answer_csv['set'] == 'train']
    test_answers = answer_csv[answer_csv['set'] == 'test']

    compute_train_hallucination_rate(train_answers)
    compute_test_accuracy(test_answers)
    compute_test_hallucination_rate(test_answers)
    compute_test_answer_length_distribution(test_answers)


if __name__ == '__main__':
    main()
