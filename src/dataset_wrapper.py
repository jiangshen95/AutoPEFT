'''Get dataset and preprocess
split the dataset into train and test
'''

from datasets import load_dataset


class PEFTDataset():
    '''
    '''
    dataset = None

    def __init__(self, dataset_name, test_size=0.8):
        '''
        dataset_name: a string
        '''
        self.dataset = load_dataset(dataset_name)
        self.dataset = self.dataset['test'].train_test_split(
            test_size=test_size)

    def get_dataset(self):
        return self.dataset
