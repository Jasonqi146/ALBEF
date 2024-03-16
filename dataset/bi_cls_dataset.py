from torch.utils.data import Dataset
from dataset.utils import pre_caption


class bi_cls_dataset(Dataset):
    def __init__(self, dataset, transform, max_words=1000):
        self.dataset = dataset
        self.transform = transform
        self.max_words = max_words
        self.labels = {'A':1, 'B':0}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        data_point = self.dataset[index]
        image = self.transform(data_point['image'])
        sentence = pre_caption(data_point['from_description'], self.max_words)
        
        return image, sentence, self.labels[data_point['label']]
    