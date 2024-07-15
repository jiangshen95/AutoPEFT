'''Get dataset and preprocess
split the dataset into train and test
'''

from datasets import load_dataset, DatasetDict
from torch.utils.data import random_split


class PEFTDataset():
    '''
    '''
    dataset = None
    train_dataset = None
    test_dataset = None
    validation_dataset = None

    def __init__(self,
                 dataset_name,
                 task_name=None,
                 instructs=False,
                 test_size=1.0,
                 train_size=1.0):
        '''
        dataset_name: a string,
        task_name: a string, can be empty, only use in "glue/cola" etc.
        '''
        if (task_name):
            dataset = load_dataset(dataset_name, task_name)
        else:
            dataset = load_dataset(dataset_name)

        dataset = dataset.shuffle(seed=42)

        instruct_string = ""
        if (dataset_name == "glue"):
            if (task_name == "cola"):
                pass

        # split the dataset, dataset should have "train", "test", "validation"
        # if train_size/test_size<1.0, then the dataset will be split by rate
        # if train_size/test_size>1.0, then the dataset will be split by number

        original_train_size = len(dataset['train'])
        original_val_size = len(dataset['validation'])
        original_test_size = len(dataset['test'])

        if train_size <= 1.0:
            new_train_size = int(original_train_size * train_size)
        elif train_size > 1.0:
            new_train_size = min(int(train_size), original_train_size)

        if test_size <= 1.0:
            new_val_size = int(original_val_size * test_size)
            new_test_size = int(original_test_size * test_size)
        elif test_size > 1.0:
            new_val_size = min(int(test_size), original_val_size)
            new_test_size = min(int(test_size), original_test_size)

        train_datast, val_dataset, test_dataset = load_dataset(
            dataset_name,
            task_name,
            split=[
                f'train[:{new_train_size}]', f'validation[:{new_val_size}]',
                f'test[:{new_test_size}]'
            ])

        dataset = DatasetDict({
            'train': train_datast,
            'validation': val_dataset,
            'test': test_dataset
        })

        if instructs and not instruct_string:

            def add_prefix(example):
                example["text"] = instruct_string + example["text"]
                return example

            dataset = dataset.map(add_prefix)

        self.dataset = dataset

    def get_dataset(self):
        return self.dataset
