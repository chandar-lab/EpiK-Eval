import os
import pandas as pd
import random
import torch
import warnings


class TrainDataset(torch.utils.data.IterableDataset):
    """
    Args:
        tokenizer (transformers.PreTrainedTokenizer): Huggingface tokenizer to tokenize samples.
        model_type (str): 'causal' or 't5'. Default: 'causal'
        segmented_docs (bool): Whether to use segmented stories. If False, will use unsegmented stories. Default: False
        unsegmented_document_directory (str): Path to the directory containing the unsegmented stories. Default: 'data/dataset/unsegmented_documents/'
        segmented_document_directory (str): Path to the directory containing the segmented stories. Default: 'data/dataset/segmented_documents/'
        qa_examples (str): Path to the question-answer examples. Default: 'data/dataset/qa_examples.csv'
    """

    def __init__(
            self,
            tokenizer,
            model_type='causal',
            segmented_docs=False,
            unsegmented_document_directory='data/dataset/unsegmented_documents/',
            segmented_document_directory='data/dataset/segmented_documents/',
            qa_examples='data/dataset/qa_examples.csv'):
        super().__init__()
        assert model_type in ['causal', 't5'], f"Expected model_type to be either 'causal' or 't5', but got: {model_type}"
        self.model_type = model_type

        documents = []
        document_directory = segmented_document_directory if segmented_docs else unsegmented_document_directory
        for file in os.listdir(document_directory):
            if file[-4:] == '.txt':
                with open(os.path.join(document_directory, file), 'r') as f:
                    documents.append(f.read())

        qa_examples = pd.read_csv(qa_examples, dtype=str, delimiter='|')

        # Create batch once since always the same
        if model_type == 'causal':
            _input = []
            for document in documents:
                _input.append(document + tokenizer.eos_token)
            for idx, row in qa_examples.iterrows():
                _input.append(f"{row['question']}\n{row['answer']}{tokenizer.eos_token}")

            _input = tokenizer(_input, padding=True, return_tensors='pt', return_length=True)
            self.attention_mask = _input.attention_mask
            self.input_length = _input.length
            self.input = _input.input_ids
        elif model_type == 't5':
            _input = []
            target = []
            for document in documents:
                document = document.split('\n')
                _input.append(document[0])  # banner goes to encoder
                target.append('\n'.join(document[1:]))  # rest goes to decoder
            for idx, row in qa_examples.iterrows():
                _input.append(row['question'])
                target.append(row['answer'])

            _input = tokenizer(_input, padding=True, return_tensors='pt', return_length=True)
            self.attention_mask = _input.attention_mask
            self.input_length = _input.length
            self.input = _input.input_ids
            target = tokenizer(target, padding=True, return_tensors='pt', return_length=True)
            self.target_length = target.length
            self.target = target.input_ids
            self.target[self.target == tokenizer.pad_token_id] = -100  # -100 is torch loss' default ignore index

    def __iter__(self):
        # Accelerate dataloader expects dataset to yield individual samples.
        # Alternative would be to modify Accelerate's dataloader.

        random_order = list(range(self.input.shape[0]))
        random.shuffle(random_order)

        if self.model_type == 'causal':
            for i in random_order:
                yield self.input[i], self.attention_mask[i], self.input_length[i]
        elif self.model_type == 't5':
            for i in random_order:
                yield self.input[i], self.attention_mask[i], self.input_length[i], self.target[i], self.target_length[i]

    def __len__(self):
        return self.input.shape[0]


class EvaluationDataset(torch.utils.data.IterableDataset):
    """
    Args:
        tokenizer (transformers.PreTrainedTokenizer): Huggingface tokenizer to tokenize samples.
        batch_size (int): Evaluation batch size
        model_type (str): 'causal' or 't5'. Default: 'causal'
        segmented_docs (bool): Whether to use segmented stories. If False, will use unsegmented stories. Default: False
        unsegmented_document_directory (str): Path to the directory containing the unsegmented stories. Default: 'data/dataset/unsegmented_documents/'
        segmented_document_directory (str): Path to the directory containing the segmented stories. Default: 'data/dataset/segmented_documents/'
        eval_questions_csv (str): Path to the evaluation (val & test) questions. Default: 'data/dataset/eval_questions.csv'
    """

    def __init__(
            self,
            tokenizer,
            batch_size,
            model_type='causal',
            segmented_docs=False,
            unsegmented_document_directory='data/dataset/unsegmented_documents/',
            segmented_document_directory='data/dataset/segmented_documents/',
            eval_questions_csv='data/dataset/eval_questions.csv'):
        super().__init__()
        assert model_type in ['causal', 't5'], f"Expected model_type to be either 'causal' or 't5', but got: {model_type}"

        qa_df = pd.read_csv(eval_questions_csv, dtype=str, delimiter='|')
        questions = qa_df['question'].to_list()
        answers = qa_df['answer'].to_list()

        # Model is also evaluated for its ability to memorize training samples.
        train_doc_ids = []
        train_doc_tasks = []
        train_doc_banners = []
        train_doc_stories = []
        document_directory = segmented_document_directory if segmented_docs else unsegmented_document_directory
        for file in sorted(os.listdir(document_directory)):
            if file[-4:] == '.txt':
                with open(os.path.join(document_directory, file), 'r') as f:
                    doc = f.read().split('\n')
                    train_doc_ids.append(file[:4])
                    train_doc_tasks.append(qa_df.loc[qa_df['story id'] == file[:4]]['task'].item())
                    train_doc_banners.append(doc[0])
                    if model_type == 't5': # T5 tokenizer converts \n into spaces, so must replace \n with space
                        train_doc_stories.append(' '.join(doc[1:]))
                    else:
                        train_doc_stories.append('\n'.join(doc[1:]))

        self.prompts = questions + train_doc_banners
        self.targets = answers + train_doc_stories

        self.tokenized_prompts = tokenizer(self.prompts, padding=True, return_tensors='pt', return_length=True)

        train_df = pd.DataFrame({
            'set': ['train'] * len(train_doc_ids),
            'story id': train_doc_ids,
            'task': train_doc_tasks,
            'question': train_doc_banners,
            'answer': train_doc_stories
        })
        self.eval_df = pd.concat([qa_df, train_df], ignore_index=True)

    def __iter__(self):
        for i in range(len(self.prompts)):
            yield i, self.tokenized_prompts.input_ids[i], self.tokenized_prompts.attention_mask[i], self.tokenized_prompts.length[i]

    def __len__(self):
        return len(self.prompts)


class TrainDataLoader(torch.utils.data.DataLoader):
    """Wrapper for torch.utils.data.DataLoader. Is meant to be used with the TrainDataset, required by the Accelerate module, but then gets replaced with Accelerate's own dataloader."""

    def __init__(self, tokenizer, model_type, segmented_docs, batch_size):
        dataset = TrainDataset(tokenizer, model_type, segmented_docs)
        super().__init__(dataset, batch_size=batch_size, shuffle=False)

    def __iter__(self):
        for batch in self.dataset:
            yield batch


class EvaluationDataLoader(torch.utils.data.DataLoader):
    """Wrapper for torch.utils.data.DataLoader. Is meant to be used with the QuestionDataset, required by the Accelerate module, but then gets replaced with Accelerate's own dataloader."""

    def __init__(self, tokenizer, segmented_docs, batch_size, model_type):
        super().__init__(
            EvaluationDataset(
                tokenizer,
                batch_size,
                model_type,
                segmented_docs),
            batch_size=batch_size,
            shuffle=False)

    def __iter__(self):
        for batch in self.dataset:
            yield batch
