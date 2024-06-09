'''Get dataset and preprocess
split the dataset into train and test
'''

from datasets import load_dataset


class PEFTDataset():
    '''
    '''
    dataset = None

    def __init__(self, dataset_name, instructs=False, test_size=0.8):
        '''
        dataset_name: a string
        '''
        self.dataset = load_dataset(dataset_name)
        # if instructs and dataset_name == "rotten_tomatoes":
        #     instruct_string = "Below is some movie reviews that are labeled positive or negative. If it is positive, output 1, else 0. Do not output anything else. Here is the review: "
        # else:
        #     instruct_string = ""
        # for j in ['train', 'validation', 'test']:
        #     for i in range(len(self.dataset[j]["text"])):
        #         self.dataset[j]["text"][
        #             i] = instruct_string + self.dataset[j]["text"][i]

        # TODO: 试图在数据集前面添加指令，但是改不了
        self.dataset = self.dataset['test'].train_test_split(
            test_size=test_size)
        print(self.dataset)

    def get_dataset(self):
        return self.dataset
