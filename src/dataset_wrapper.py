'''Get dataset and preprocess
split the dataset into train and test
'''

from datasets import load_dataset, DatasetDict


class PEFTDataset():
    '''
    '''
    dataset = None

    def __init__(self,
                 dataset_name,
                 instructs=False,
                 test_size=1.0,
                 train_size=1.0):
        '''
        dataset_name: a string
        '''
        # dataset_train = load_dataset(
        #     'rotten_tomatoes', split=f"train[:{int(train_size*100)}%]")
        # dataset_test = load_dataset(
        #     'rotten_tomatoes', split=f"test[:{int(test_size*100)}%]")
        # self.dataset = DatasetDict({
        #     'train': dataset_train,
        #     'test': dataset_test
        # })
        self.dataset = load_dataset(dataset_name)
        self.dataset = self.dataset['test'].train_test_split(
            test_size=test_size)
        if instructs and dataset_name == "rotten_tomatoes":
            instruct_string = "Below is some movie reviews that are labeled positive or negative. If it is positive, output 1, else 0. Do not output anything else. Here is the review: \n"
        else:
            instruct_string = ""

        def add_prefix(example):
            example["text"] = instruct_string + example["text"]
            return example

        self.dataset = self.dataset.map(add_prefix)
        # print(self.dataset['train']['text'][:10])

    def get_dataset(self):
        return self.dataset
