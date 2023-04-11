import os

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from models.EncoderDataset import *
from utils.load_datasets import load_dataset

class TopSegDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", 
                 dataset = "BBC", 
                 encoder: str = "roberta_base_second_to_last_mean", 
                 architecture: str = "BiLSTM",
                 max_length: int = None,
                 batch_size: int = 8,
                 embedding_directory: str = None,
                 online_encoding = False,
                 validation_percentage: float = 0.2,
                 test_percentage: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.encoder = encoder
        self.architecture = architecture
        self.max_length = max_length
        self.valid_percentage = validation_percentage
        self.test_percentage = test_percentage 
        self.batch_size = batch_size
        self.embeddings_directory = embedding_directory
        self.online_encoding = online_encoding

    def prepare_data(self):
        # download
        self.WordMatrix = None
        try:
            self.folds = load_dataset(self.dataset) # TODO: add the additional options from load_dataset function
            if len(self.folds)>1:
                raise NotImplementedError("Cross Validation is not currently supported in this version of the project! Working on this...")
            elif len(self.folds)==1:
                folds = self.folds[0]
                assert len(folds), "The chosen dataset returned empty list! Check that you correctly implemented the dataset inclusion procedure, as explained in the README file in the utils subfolder!"
                self.has_validation = True
                self.has_test = True
                if len(folds)==3:
                    pass
                elif len(folds)==2:
                    self.has_validation = False
                elif len(folds)==1:
                    self.has_validation = False
                    self.has_test = False
                else:
                    raise ValueError("Your chosen dataset returned either too many splits (>3) or no split at all.")
            else:
                raise ValueError("The chosen dataset returned empty list! Check that you correctly implemented the dataset inclusion procedure, as explained in the README file in the utils subfolder!")
        
            if self.architecture.lower()=="textseg":
                import gensim.downloader
                
                self.WordMatrix = gensim.downloader.load('word2vec-google-news-300')
                
                self.word2index = {w:i for i, w in enumerate(WordMatrix.index_to_key)}
                
                self.WordMatrix = createPreTrainedEmbedding(WordMatrix, word2index, False)

        except ValueError:
            raise ValueError("The name of the dataset was not recognised! Did you add a condition in the load_dataset function or did you pass a custom function to the dataset argument to return your dataset? If not, follow the rules in the README file inside the utils subfolder to use your custom dataset.")
        

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders

        if self.architecture.lower().startswith('text'):
            encoder = None
        elif self.architecture.lower().startswith('crossencoder'):
            encoder, _ = self.encoder.split("+")
        else:
            encoder = get_model(self.encoder, max_length = self.max_length)[0] # TODO: add the additional options from get_model function
        
            in_dim = encoder.get_sentence_embedding_dimension()

        precompute = not self.online_encoding
            
        CRF = True if self.architecture.lower().endswith('crf') else False

        tag_to_ix = {0:0, 1:1, '<START>':2, '<STOP>':3}

        if stage == "fit":
            
            if self.architecture.lower().startswith('text'):
                
                for fold in self.folds:

                    if self.has_validation:
                        train_samples = fold[0]
                        valid_samples = fold[2]
                    else:
                        print('performing train/validation split')
                        valid_num = int(self.valid_percentage*100)
                        train_samples = []
                        valid_samples = []
                        for index, sample in enumerate(fold[0]):
                            if index%valid_num==0:
                                valid_samples.append(sample)
                            else:
                                train_samples.append(sample)

                    self.train_dataset = WordLevelDataset(train_samples, tag_to_ix, self.word2index, self.WordMatrix)
                    self.valid_dataset = WordLevelDataset(valid_samples, tag_to_ix, self.word2index, self.WordMatrix)
            
            elif self.architecture.lower().startswith('crossencoder'):
                for fold in self.folds:
                    if self.has_validation:
                        train_samples = fold[0]
                        valid_samples = fold[2]
                    else:
                        print('performing train/validation split')
                        valid_num = int(self.valid_percentage*100)
                        train_samples = []
                        valid_samples = []
                        for index, sample in enumerate(fold[0]):
                            if index%valid_num==0:
                                valid_samples.append(sample)
                            else:
                                train_samples.append(sample)

                    self.train_dataset = CrossEncoderDataset(train_samples, enc=encoder, minus = CRF, longformer = False)
                    self.valid_dataset = CrossEncoderDataset(valid_samples, enc=encoder, minus = CRF, longformer = False)
            
            elif self.embeddings_directory is None:
                for fold in self.folds:
                    if self.has_validation:
                        train_samples = fold[0]
                        valid_samples = fold[2]
                    else:
                        print('performing train/validation split')
                        valid_num = int(self.valid_percentage*100)
                        train_samples = []
                        valid_samples = []
                        for index, sample in enumerate(fold[0]):
                            if index%valid_num==0:
                                valid_samples.append(sample)
                            else:
                                train_samples.append(sample)

                    # TODO: Add additional options for SentenceDataset among the ones available
                    self.train_dataset = SentenceDataset(train_samples, tag_to_ix, encoder = encoder, precompute = precompute, CRF =CRF)
                    self.valid_dataset = SentenceDataset(valid_samples, tag_to_ix, encoder = encoder, precompute = precompute, CRF =CRF)
            else:
                if self.has_validation:
                    train_embeddings = load_embeddings(self.encoder + '_train', self.dataset, self.embeddings_directory)
                    valid_embeddings = load_embeddings(self.encoder + '_valid', self.dataset, self.embeddings_directory)
                    
                    for fold in self.folds:
                        self.train_dataset = SentenceDataset(fold[0], tag_to_ix, encoder = encoder, embeddings = train_embeddings, CRF =CRF)
                        self.valid_dataset = SentenceDataset(fold[2], tag_to_ix, encoder = encoder, embeddings = valid_embeddings, CRF =CRF)
                else:
                    train_embeddings = load_embeddings(self.encoder + '_train', self.dataset, self.embeddings_directory)

                    valid_num = int(self.valid_percentage*100)
                    train_samples = []
                    valid_samples = []
                    train_embeddings_new = []
                    valid_embeddings = []

                    for index, sample in enumerate(fold[0]):
                        if index%valid_num==0:
                            valid_samples.append(sample)
                            valid_embeddings.append(train_embeddings[index])
                        else:
                            train_samples.append(sample)
                            train_embeddings_new.append(train_embeddings[index])
                    
                    for fold in self.folds:
                        self.train_dataset = SentenceDataset(train_samples, tag_to_ix, encoder = encoder, embeddings = train_embeddings_new, CRF =CRF)
                        self.valid_dataset = SentenceDataset(valid_samples, tag_to_ix, encoder = encoder, embeddings = valid_embeddings, CRF =CRF)
        if stage == "test":
            if self.architecture.lower().startswith('text'):
                
                import gensim.downloader
                
                WordMatrix = gensim.downloader.load('word2vec-google-news-300')
                
                word2index = {w:i for i, w in enumerate(WordMatrix.index_to_key)}
                
                WordMatrix = createPreTrainedEmbedding(WordMatrix, word2index, False)

                for fold in self.folds:

                    if self.has_test:
                        train_samples = fold[0]
                        test_samples = fold[1]
                    else:
                        print('performing train/validation split')
                        valid_num = int(self.test_percentage*100)
                        train_samples = []
                        valid_samples = []
                        for index, sample in enumerate(fold[0]):
                            if index%valid_num==0:
                                test_samples.append(sample)

                    self.test_dataset = WordLevelDataset(test_samples, tag_to_ix, word2index, WordMatrix)
            elif self.architecture.lower().startswith('crossencoder'):
                for fold in self.folds:

                    if self.has_test:
                        train_samples = fold[0]
                        test_samples = fold[1]
                    else:
                        print('performing train/validation split')
                        valid_num = int(self.test_percentage*100)
                        train_samples = []
                        valid_samples = []
                        for index, sample in enumerate(fold[0]):
                            if index%valid_num==0:
                                test_samples.append(sample)
                    
                    self.test_dataset = CrossEncoderDataset(test_samples, enc=encoder, minus = CRF, longformer = False)
            
            elif self.embeddings_directory is None:
                for fold in self.folds:

                    if self.has_test:
                        train_samples = fold[0]
                        test_samples = fold[1]
                    else:
                        print('performing train/validation split')
                        valid_num = int(self.test_percentage*100)
                        train_samples = []
                        valid_samples = []
                        for index, sample in enumerate(fold[0]):
                            if index%valid_num==0:
                                test_samples.append(sample)
                
                self.test_dataset = SentenceDataset(test_samples, tag_to_ix, encoder = encoder, precompute = precompute, CRF =CRF)
            
            else:
                if self.has_test:
                    if self.has_validation:
                        test_embeddings = load_embeddings(self.encoder + '_test', self.dataset, self.embeddings_directory)
                        
                        for fold in self.folds:
                            self.test_dataset = SentenceDataset(fold[1], tag_to_ix, encoder = encoder, embeddings = test_embeddings, CRF =CRF)
                    else:
                        train_embeddings = load_embeddings(self.encoder + '_train', self.dataset, self.embeddings_directory)
                        
                        for fold in self.folds:
                            valid_num = int(self.valid_percentage*100)
                            train_samples = []
                            test_samples = []
                            train_embeddings_new = []
                            test_embeddings = []

                            for index, sample in enumerate(fold[0]):
                                if index%valid_num==0:
                                    test_samples.append(sample)
                                    test_embeddings.append(train_embeddings[index])

                            self.test_dataset = SentenceDataset(test_samples, tag_to_ix, encoder = encoder, embeddings = test_embeddings, CRF =CRF)


        if stage == "predict":
            raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.train_dataset.collater)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, collate_fn=self.valid_dataset.collater)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.test_dataset.collater)

    def predict_dataloader(self):
        raise NotImplementedError()