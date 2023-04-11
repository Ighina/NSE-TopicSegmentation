# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:37:48 2021

@author: Iacopo
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
#import tensorflow_hub as hub
from torch.utils.data import Dataset, DataLoader
from models.lightning_model import *
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import re
import time
import string
import os
from transformers import AutoTokenizer, AutoModel
#from simcse import SimCSE


nltk.download('punkt')

def load_embeddings(encoder_name, dataset_name, parent_directory = 'embeddings'):
        in_dir = os.path.join(parent_directory, dataset_name, encoder_name)
        
        assert os.path.exists(in_dir), "The directory provided {} for the embeddings does not exist!".format(in_dir)
        
        for root, dirs, files in os.walk(in_dir):
            embeddings = [0 for i in range(len(files))]
            for file in files:
                try:
                    file_num = re.findall('\w+_(\d+)\.npy', file)[0]
                except IndexError:
                    continue
                embeddings[int(file_num)] = np.load(os.path.join(root, file))
        
        embeddings = [embedding for embedding in embeddings if not isinstance(embedding, int)]
        
        return embeddings

class SentenceDataset(Dataset):
    def __init__(self, lines, tag_to_ix, encoder = 'stsb-bert-base', 
                precompute = True, embeddings = None, 
                infer = False, CRF = True, cosine_loss = False, 
                manual_max_length = None, mask_inner_sentences = False,
                mask_probability = 0.7, second_encoder = None):
        
        self.cos = cosine_loss
        self.infer = infer
        self.minus = 0 if CRF else 1
        self.max_length = manual_max_length
        
        self.mi = mask_inner_sentences
        self.mp = mask_probability
        
        if infer:
            self.sentences = lines
            self.tgt_dataset = [0 for x in lines]
        else:
            self.sentences = [line[0] for line in lines]
            self.tgt_dataset = [[tag_to_ix[tag] for tag in line[1]] for line in lines]
        if isinstance(encoder, str):
            self.encoder = SentenceTransformer(encoder)
        else:
            self.encoder = encoder #re-use an elsewhere defined encoder
        if second_encoder is not None:
            if isinstance(second_encoder, str):
                self.encoder2 = SentenceTransformer(second_encoder)
            else:
                self.encoder2 = second_encoder #re-use an elsewhere defined encoder
        else:
            self.encoder2 = None
        if embeddings is not None:
            self.precompute = True
            self.embeddings = [torch.tensor(embedding) for embedding in embeddings]
        
        else:    
            if precompute: # TODO: include multiprocessing for generating the embeddings (with gpu/cpu support)
                self.precompute = True
                self.embeddings = [self.encoder.encode(sents, convert_to_tensor = True).detach().cpu() for sents in self.sentences]
                if second_encoder is not None:
                    second_embeddings = [self.encoder2.encode(sents, convert_to_tensor = True, show_progress_bar = False).detach().cpu() for sents in self.sentences]
                    print("Finished encoding with the second encoder!")
                    self.encoder2 = None # if pre-computing, it is useless to keep the encoder in the dataset object after encoding is done
                    self.embeddings = [torch.cat((emb, second_embeddings[i]), axis = 1) for i, emb in enumerate(self.embeddings)]
                self.encoder = None # if pre-computing, it is useless to keep the encoder in the dataset object after encoding is done
                
            else:
                self.precompute = False
                self.embeddings = [0 for x in range(len(self.sentences))]
        
    @classmethod
    def from_precomputed(cls, lines, tag_to_ix, encoder_name, dataset_name, parent_directory = 'embeddings'):
        embeddings = load_embeddings(encoder_name, dataset_name, parent_directory)
        return cls(lines = lines, tag_to_ix = tag_to_ix, 
                    encoder = encoder_name, precompute = True, 
                    embeddings = embeddings)
    
    def __getitem__(self, index):
        return {
            'id': torch.tensor(index),
            'source': self.sentences[index],
            'target': self.tgt_dataset[index],
            'embeddings': self.embeddings[index]
        }
        
    def __len__(self):
        return len(self.sentences)
        
    def save_embeddings(self, encoder_name, dataset_name, parent_directory = 'embeddings'):
        assert len(self.embeddings[0])>1, 'The embeddings need to be precomputed in order to be saved'
        
        out_dir = os.path.join(parent_directory, dataset_name, encoder_name)
        
        if os.path.exists(out_dir):
            assert len(os.listdir(out_dir))==0, 'The folder where to save the embeddings is not empty: if you want to save your embeddings first make sure that the folder {} is empty'.format(out_dir)
        else:
            os.makedirs(out_dir)
            
        for i in range(self.__len__()):
            try:
                np.save(os.path.join(out_dir, 'embeddings_'+str(i)), self.embeddings[i])
            except TypeError:
                np.save(os.path.join(out_dir, 'embeddings_'+str(i)), self.embeddings[i].detach().cpu())
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        def merge(values, continuous=False):
            if len(values[0].shape)<2:
              return torch.stack(values)
            else:
              if self.max_length is None:
                max_length = max(v.size(0) for v in values)
              else:
                max_length = self.max_length
              result = torch.zeros((len(values),max_length, values[0].shape[1]))
              for i, v in enumerate(values):
                  result[i, :len(v)] = v
              return result

        def merge_tags(tags):
          if self.max_length is None:
            max_length = max(v.size(0) for v in tags)
          else:
            max_length = self.max_length
          result = torch.zeros((len(tags),max_length)) - self.minus
          for i, v in enumerate(tags):
              if self.mi:
                  for index in range(len(v)):
                    if np.random.rand()<self.mp and not v[index]:
                      v[index] = v[index]-self.minus # avoiding using the mask where the ignore index argument is not allowed (e.g. BinaryCrossEntropy loss and Conditional Random Fields models)
                      
              result[i, :len(v)] = v
          return result
          
        
        if self.precompute:
            id = torch.tensor([s['id'] for s in samples])
            src_tokens = merge([s['embeddings'] for s in samples])
            if self.infer:
                tgt_tokens = None
            else:
                tgt_tokens = merge_tags([torch.tensor(s['target']) for s in samples])
                
            if self.cos:
                segments = [[index for index, x in enumerate(s['target']) if x] for s in samples]
            else:
                segments = None
            
            src_lengths = torch.LongTensor([len(s['source']) for s in samples])
            src_sentences = [s['source'] for s in samples]
        else:
            id = torch.tensor([s['id'] for s in samples])
            if self.encoder2 is not None:
                src_tokens = merge([torch.cat((self.encoder.encode(s['source'], convert_to_tensor = True), self.encoder2.encode(s['source'], convert_to_tensor = True))) for s in samples])
            else:
                src_tokens = merge([self.encoder.encode(s['source'], convert_to_tensor = True) for s in samples])
            if self.infer:
                tgt_tokens = None
            else:
                tgt_tokens = merge_tags([torch.tensor(s['target']) for s in samples])
                
            if self.cos:
                segments = [[index for index, x in enumerate(s['target']) if x] for s in samples]
            else:
                segments = None
            
            src_lengths = torch.LongTensor([len(s['source']) for s in samples])
            src_sentences = [s['source'] for s in samples]
            

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_tokens': tgt_tokens,
            'src_sentences': src_sentences,
            'src_segments': segments
        }

class CrossEncoderDataset(Dataset):
    def __init__(self, all_sentences, enc = "bert-base-uncased", max_length = 512, minus = False, longformer = False):
        self.tokenizer = AutoTokenizer.from_pretrained(enc)
        sep_token = self.tokenizer._sep_token
        if longformer:
            self.sentences = all_sentences
        else:
            self.sentences = [[f' {sep_token} '.join([sentences[0][i], sentences[0][i + 1]]) for i in range(len(sentences[0]) - 1)] + [sentences[0][-1]] for sentences in all_sentences]
        self.labels = [sentences[1] for sentences in all_sentences]
        
        self.max_length = max_length
        self.pad_id = self.tokenizer._pad_token_type_id
        self.mi = 1 if minus else 0

    def __getitem__(self, index):
        return {
            'id': torch.tensor(index),
            'source': self.sentences[index],
            'target': self.labels[index]
        }
        
    def __len__(self):
        return len(self.sentences)
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        
        def merge(tags, input_tensor = True, labels = False):
            if self.max_length is None:
                max_length = max(len(v) for v in tags)
            else:
                max_length = self.max_length
            data_type = torch.float32 if labels else torch.int64 
            if input_tensor:
                if labels:
                    result = torch.zeros((len(tags),max_length), dtype=data_type) - self.mi
                    
                    for i, v in enumerate(tags):
                        result[i, :len(v)] = v
                    
                else:
                    max_tokens = max([tag.size(1) for tag in tags])
                    
                    result = torch.zeros((len(tags),max_length,max_tokens), dtype=data_type) + self.pad_id
                    
                    for i, v in enumerate(tags):
                        result[i, :len(v), :v.size(1)] = v
                    
            else:
                return [inp[:max_length] if len(inp)>=max_length else inp+[0 for x in range(max_length-len(inp))] for inp in tags]
            
            return result
        
        src_sentences = [s["source"] for s in samples]
        for s in src_sentences:
            if len(s)<1:
                print("This is the problem:")
                0/0
        try:
            tokens_and_masks = [self.tokenizer(s["source"], padding=True, truncation=True, return_tensors='pt', max_length=self.max_length) for s in samples]
        except IndexError:
            print("this did not work: here are the sentences:")
            print(src_sentences)
            0/0
        src_tokens = merge([t["input_ids"] for t in tokens_and_masks])
        src_attention_masks = merge([t["attention_mask"] for t in tokens_and_masks])
        
        src_lengths = torch.LongTensor([len(s['source']) for s in samples])
        tgt_tokens = merge([torch.tensor(s['target']) for s in samples], labels = True)
        segments = None

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_attention_mask': src_attention_masks,
            'tgt_tokens': tgt_tokens,
            'src_sentences': src_sentences,
            'src_segments': segments
        }

class Predictor:
    def __init__(self, trained_model, sentence_encoder, tag2ix = None, remove_char = None):
        self.model = trained_model
        if isinstance(sentence_encoder, str):
            self.se = SentenceTransformer(sentence_encoder)
        else:
            self.se = sentence_encoder
        self.remove_char = remove_char
        if tag2ix is None:
            self.tags = {0:0, 1:1, '<START>':2, '<STOP>':3}
        else:
            self.tags = tag2ix
    
    def online_sents_encoder(self, input_sentences):
        embs = []
        for sentence in input_sentences:
            embs.append(self.se.encode(sentence))
        
        return torch.tensor(embs).unsqueeze(0), torch.LongTensor([len(embs)])
    
    def preprocess(self, doc, delete_special_char = True, 
                   delete_punct = False, just_words = False):
        sentences = nltk.sent_tokenize(doc)
        
        input_sentences = []
        
        for sentence in sentences:
            if just_words:
                input_sentence = re.sub('[^a-z\s]', '', sentence).strip()
            elif delete_punct:
                input_sentence = re.sub('[^\w\s]', '', sentence)
            elif delete_special_char:
                if self.remove_char is None:
                    input_sentence = re.sub('[\=#@{\|}~\[\]\^_]', '', sentence)
                else:
                    input_sentence = re.sub(self.remove_char, '', sentence)
            
            if input_sentence:
                input_sentences.append(input_sentence)
        
        return input_sentences
    
    def predict(self, docs, batch = False, 
                delete_special_char = True, 
                delete_punct = False, 
                just_words = False,
                pretokenized_sents = None,
                device = None,
                verbose = False):
        
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else: 
                device = 'cpu'
        
        results = []
        if batch:
            
            
            
            batch_size = min(100, len(docs))
            
            if verbose:
                print('Aggregating {} documents together to process them in batch...'.format(batch_size))
            
            tokenized_sents = []
            
            for index, doc in enumerate(docs):
                if pretokenized_sents:
                    tokenized_sents.append(pretokenized_sents[index])
                else:
                    tokenized_sents.append(self.preprocess(doc, 
                                                           delete_special_char, 
                                                           delete_punct, 
                                                           just_words))
                
                
                
            InferDataset = SentenceDataset(tokenized_sents, 
                                           self.tags, 
                                           encoder = self.se,
                                           precompute=True,
                                           infer = True)
            
            dl = DataLoader(InferDataset, batch_size = batch_size, 
                            collate_fn = InferDataset.collater)
            
            results = []
            
            for index, batch in enumerate(dl):
                if verbose:
                    print('Segmenting batch number {}...'.format(index))
                
                inputs = batch['src_tokens'].to(device)
                lengths = batch['src_lengths'].to(device)
                input_sentences = batch['src_sentences']
                
                batch_scores, batch_boundaries = self.model.model(inputs, lengths)
                
                if verbose:
                    print('Batch number {} segmented.'.format(index))
                
                for batch_index, boundaries in enumerate(batch_boundaries):
                    
                    embs = inputs[batch_index][:lengths[batch_index]].squeeze().detach().cpu().numpy()
                    
                    
                    segments = []
                    segmented_embs = []
                    last_index = 0
                    for index, boundary in enumerate(boundaries):
                        if boundary:
                            segments.append(input_sentences[batch_index][last_index:index + 1])
                            segmented_embs.append(embs[last_index:index + 1])
                            last_index = index + 1
                    
                    segments.append(input_sentences[batch_index][last_index:])
                    
                    results.append({'segments': segments, 
                        'boundaries': boundaries,
                        'scores': batch_scores[batch_index],
                        'embeddings': segmented_embs})
            
        else:
            for index, doc in enumerate(docs):
                if pretokenized_sents is None:
                    input_sentences = self.preprocess(doc, 
                                                       delete_special_char, 
                                                       delete_punct, 
                                                       just_words)
                else:
                    input_sentences = pretokenized_sents[index]
                
                inputs, length = self.online_sents_encoder(input_sentences)
                
                scores, boundaries = self.model.model(inputs, length)
                
                embs = inputs.squeeze().numpy()
                
                segments = []
                segmented_embs = []
                last_index = 0
                for index, boundary in enumerate(boundaries[0]):
                    if boundary:
                        segments.append(input_sentences[last_index:index + 1])
                        segmented_embs.append(embs[last_index:index + 1])
                        last_index = index + 1
                
                segments.append(input_sentences[last_index:])
                
                results.append({'segments': segments, 
                    'boundaries': boundaries,
                    'scores': scores,
                    'embeddings': segmented_embs})
        
        return results
    
def save_segmentation_results(results, output_directory, results_directory = 'results'):
    save_dir = os.path.join(results_directory, output_directory)
    
    assert not os.path.exists(save_dir)
    
    os.makedirs(os.path.join(save_dir, 'segments'))
    
    os.mkdir(os.path.join(save_dir, 'embeddings'))


class BERT_BASE_ENCODER:
    def __init__(self, enc, pool, opt=None, emb_diff=False, max_length = 512):
        self.bert = AutoModel.from_pretrained(enc)
        self.tokenizer = AutoTokenizer.from_pretrained(enc)
        self.model = 'bert_cls_token'
        self.max_length = max_length
        if pool == "cls":
            self.pool = self.cls_pooling
        elif pool == "last_first_mean":
            self.pool = self.first_last_pooling
        elif pool == "last_mean":
            self.pool = self.last_pooling
        elif pool == "second_to_last_mean":
            self.pool = self.second_to_last_pooling  # the default in BERT as a service
        elif pool == "sep":
            self.pool = self.sep_pooling
        else:
            raise ValueError("Pooling strategy not recognised!")

        if not opt in ("pairwise", "combined"):
            print(
                "Warning: the optional configuration of using pairwise or combined encoding has not been properly formatted. If you wanted to use one of those two options you should have attached +pairwise/+combined to your model name. For now, the program will default to the usual encoding (single sentences encoding)!")

        self.opt = opt

        self.diff = False
        if emb_diff:
            self.diff = True

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.bert.to('cuda')
            print('Moved model to gpu!')
        else:
            print('No gpu is being used')
            self.device = 'cpu'

    def cls_pooling(self, model_output, attention_mask):
        return model_output[-1][:, 0, :]

    def first_last_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output[-1].size()).float()

        last = torch.sum(model_output[-1] * input_mask_expanded, dim=1)
        first = torch.sum(model_output[1] * input_mask_expanded, dim=1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return (last + first) / sum_mask

    def last_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output[-1].size()).float()
        sum_embeddings = torch.sum(model_output[-1] * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask

    def second_to_last_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output[-2].size()).float()
        sum_embeddings = torch.sum(model_output[-2] * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask

    def sep_pooling(self, model_output, attention_mask):
        return model_output[-1][:, -1, :]

    def encode(self, sentences, convert_to_tensor=False, batch_size=32):

        if self.opt == "pairwise":
            sentences = [' [SEP] '.join([sentences[i], sentences[i + 1]]) for i in
                         range(len(sentences) - 1)] + [sentences[-1]]

        all_embeddings = []

        length_sorted_idx = np.argsort([-len(sen.split()) for sen in sentences])

        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):

            sentences_batch = sentences_sorted[start_index:start_index + batch_size]

            encoded_input = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt',
                                           max_length=self.max_length)

            with torch.no_grad():

                if encoded_input['input_ids'].shape[0] > 100:
                    pass

                try:
                    model_output = self.bert(input_ids=encoded_input['input_ids'].to(self.device),
                                         attention_mask=encoded_input['attention_mask'].to(self.device),
                                         output_hidden_states=True).hidden_states
                except RuntimeError:
                    self.bert.to(self.device)
                    model_output = self.bert(input_ids=encoded_input['input_ids'].to(self.device),
                                             attention_mask=encoded_input['attention_mask'].to(self.device),
                                             output_hidden_states=True).hidden_states

                model_output = self.pool(model_output,
                                         encoded_input['attention_mask'].to(self.device)).detach().cpu()

                all_embeddings.extend(model_output)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if self.opt == "combined":
            all_embeddings_pr = []

            sentences_pr = [' [SEP] '.join([sentences[i], sentences[i + 1]]) for i in
                            range(len(sentences) - 1)] + [sentences[-1]]

            length_sorted_idx_ = np.argsort([-len(sen.split()) for sen in sentences_pr])

            sentences_sorted = [sentences_pr[idx] for idx in length_sorted_idx]

            for start_index in range(0, len(sentences), batch_size):

                sentences_batch = sentences_sorted[start_index:start_index + batch_size]

                encoded_input = self.tokenizer(sentences_batch, padding=True, truncation=True,
                                               return_tensors='pt', max_length=self.max_length)

                with torch.no_grad():

                    if encoded_input['input_ids'].shape[0] > 100:
                        pass

                    model_output = self.bert(input_ids=encoded_input['input_ids'].to(self.device),
                                             attention_mask=encoded_input['attention_mask'].to(self.device))

                    model_output = self.cls_pooling(model_output, encoded_input['attention_mask'].to(
                        self.device)).detach().cpu()

                    all_embeddings_pr.extend(model_output)

            all_embeddings_pr = [all_embeddings_pr[idx] for idx in np.argsort(length_sorted_idx)]

            if convert_to_tensor:
                # if all_embeddings_pr.shape[0]!=all_embeddings.shape[0]:
                #     print(all_embeddings.shape)
                #     print(all_embeddings_pr.shape[0])
                #     raise ValueError()
                return torch.cat((torch.stack(all_embeddings), torch.stack(all_embeddings_pr)), axis=1)
            else:
                return np.concatenate((np.asarray([emb.numpy() for emb in all_embeddings]),
                                       np.asarray([emb.numpy() for emb in all_embeddings_pr])), axis=1)

        if convert_to_tensor:
            x = torch.stack(all_embeddings)
            if self.diff:
                return torch.cat((torch.diff(x, axis=0), x[-1].unsqueeze(0)), axis=0)
            return x
        else:
            x = np.asarray([emb.numpy() for emb in all_embeddings])
            if self.diff:
                return np.concatenate((np.diff(x, axis=0), x[-1].reshape(1, -1)))
            return x

    def get_sentence_embedding_dimension(self):
        return 768

class ZERO_model():
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

        self.mapping = {k: v for v, k in enumerate(labels)}

    def encode(self, sentences, convert_to_tensor=False):
        scores = []

        results = self.model(sentences, self.labels)
        for result in results:
            score = [0 for x in range(len(self.labels))]
            for index, lab in enumerate(result['labels']):
                score[self.mapping[lab]] = result['scores'][index]
            scores.append(score)
        if convert_to_tensor:
            return torch.tensor(scores)
        else:
            return np.array(scores)

    def get_sentence_embedding_dimension(self):
        return len(self.labels)

class USE_ENCODER():
    def __init__(self, model, emb_diff=False):
        self.model = model
        self.diff = False

        if emb_diff:
            self.diff = True

    def encode(self, sentences, convert_to_tensor=False):
        x = self.model(sentences).numpy()
        if convert_to_tensor:
            if self.diff:
                return torch.from_numpy(np.concatenate((np.diff(x, axis=0), x[-1].reshape(1, -1))))
            else:
                return torch.from_numpy(x)
        else:
            if self.diff:
                return np.concatenate((np.diff(x, axis=0), x[-1].reshape(1, -1)))
            return x

    def get_sentence_embedding_dimension(self):
        return 512

class SIMCSE_ENCODER:
    def __init__(self, model="princeton-nlp/sup-simcse-roberta-base", emb_diff=False):
        self.model = SimCSE(model)
        self.diff = False
        if emb_diff:
            self.diff = True

    def encode(self, sentences, convert_to_tensor=False):
        embs = self.model.encode(sentences)
        if convert_to_tensor:
            if self.diff:
                return torch.cat((torch.diff(embs, axis=0), embs[-1].unsqueeze(0)), axis=0)
            return embs
        else:
            if self.diff:
                return np.concatenate((np.diff(embs.detach().cpu().numpy(), axis=0),
                                       embs[-1].unsqueeze(0).detach().cpu().numpy()))
            return embs.detach().cpu().numpy()

    def get_sentence_embedding_dimension(self):
        return 768

class INFERSENT_ENCODER():
    def __init__(self, model, emb_diff=False):
        self.model = model
        self.diff = False
        if emb_diff:
            self.diff = True

    def encode(self, sentences, convert_to_tensor=False):
        x = self.model.encode(sentences, tokenize=True)

        if convert_to_tensor:
            if self.diff:
                return torch.from_numpy(np.concatenate((np.diff(x, axis=0), x[-1].reshape(1, -1))))
            else:
                return torch.from_numpy(x)
        else:
            if self.diff:
                return np.concatenate((np.diff(x, axis=0), x[-1].reshape(1, -1)))
            return x

    def get_sentence_embedding_dimension(self):
        return 4096

def get_model(model_name, max_length = None, labels = None, emb_diff = False):
    
    huggingface_models = {"distilbert-news":"andi611/distilbert-base-uncased-agnews", "roberta_base":"roberta-base",
                            "bert_base_cased":"bert-base-cased", "bert_base_uncased":"bert-base-uncased",
                                "ner":"dslim/bert-base-NER"} # supported huggingface models
    
    if model_name.lower()=='use_large':
        """
        The large version of universal sentence encoder is just a transformer architecture pre-trained in the same
        way as the smaller, more portable DAN architecture.
        """
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        use = hub.load(module_url)
        print ("module %s loaded" % module_url)
            
        return (USE_ENCODER(use, emb_diff), None)
    
    elif model_name.startswith('https') or model_name.lower().startswith('universal') or model_name.lower()=='use':
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        use = hub.load(module_url)
        print ("module %s loaded" % module_url)
            
        return (USE_ENCODER(use, emb_diff), None)
    
    elif model_name.startswith('zero_'):
        from transformers import pipeline
        assert labels is not None, "To use the zero-shot approach provide possible topic labels"
        model = model_name.split('_')[1]
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("zero-shot-classification", model=model, device = device)
            
        return (ZERO_model(classifier, labels), None)   

    
    elif model_name.startswith('simcse'):
        
        return (SIMCSE_ENCODER(emb_diff = emb_diff), None)
    
    elif model_name.startswith('infersent'):
        """
        Here, we use the pre-trained GloVe version of infersent (V1 from the official implementation).
        For any doubts concerning the files to download to use this encoder read the instructions at 
        https://github.com/facebookresearch/InferSent
        """
        V = 1
        MODEL_PATH = 'pretrained_encoders/infersent%s.pkl' % V
        assert os.path.exists(MODEL_PATH), "You need to download the pre-trained infersent model in order to use it!"
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(MODEL_PATH))
        
        W2V_PATH = 'GloVe/glove.840B.300d.txt'
        assert os.path.exists(W2V_PATH), "You need to download the relative GloVe embeddings as specified in the instructions and place the resulting folder in the main directory in order to use this encoder!"
        infersent.set_w2v_path(W2V_PATH)
        
        infersent.build_vocab_k_words(K=100000)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        infersent.to(device)

        return (INFERSENT_ENCODER(infersent, emb_diff = emb_diff), None)
    
    elif model_name.startswith("roberta_topseg"):
        if max_length is not None:
            encoder = SentenceTransformer(os.path.join("pretrained_encoders", model_name))
            encoder.max_seq_length = max_length
            
            return (encoder, None)
            
        else:
            return (SentenceTransformer(os.path.join("pretrained_encoders", model_name)), None)
    
    elif re.findall("topseg", model_name):
        if max_length is not None:
            encoder = SentenceTransformer(os.path.join("pretrained_encoders", model_name))
            encoder.max_seq_length = max_length
            
            return (encoder, None)
            
        else:
            return (SentenceTransformer(os.path.join("pretrained_encoders", model_name)), None)
    
    elif model_name.split("-")[0] in huggingface_models:
        
        enc, pool = model_name.split("-") # The model name for huggingface should be in the format {base_model_name}-{pooling_code}; see the huggingface_models dictionary.
        
        pool = pool.split("+") # the syntax for adding the option of performing combined or pairwise encoding as in Transformer over transformer is included by adding a + pairwise/combined after the pooling strategy in the model name (for now works just with huggingface models)
        
        if len(pool)==2:
            pool, opt = pool
        elif len(pool)==1:
            pool = pool[0]
            opt = None
        else:
            raise ValueError("You added too many + signs in the model name: the + sign is used to use one of pairwise or combined encoding. Please refrain from using it unless you want one (and only one) of those options!")
            
        
        enc = huggingface_models[enc]

        if max_length is None:
            max_length = 512
        
        encoder = BERT_BASE_ENCODER(enc, pool, opt=opt, emb_diff = emb_diff, max_length=max_length)
        return (encoder, None)
    
    else:
        if max_length is not None:
            encoder = SentenceTransformer(model_name)
            encoder.max_seq_length = max_length
            
            return (encoder, None)
            
        else:
            return (SentenceTransformer(model_name), None)

class TopBertEncoder():
    """
    We simply call the pre-existent functions to initialise both sentence encoder and vector quantised encoder.
    In the encode function, we concatenate the two embeddings along axis 1.
    """

    def __init__(self, model_path, model_name, max_length, labels):
        self.top_encoder = get_vq_model(model_path)
        self.bert_encoder = get_model(model_name, max_length, labels)[0]

    def get_sentence_embedding_dimension(self):
        return self.top_encoder.get_sentence_embedding_dimension() + self.bert_encoder.get_sentence_embedding_dimension()

    def encode(self, sentences, convert_to_tensor=True):
        top_emb = self.top_encoder.encode(sentences, convert_to_tensor=convert_to_tensor)
        bert_emb = self.bert_encoder.encode(sentences, convert_to_tensor=convert_to_tensor)

        if convert_to_tensor:
            top_emb = top_emb.to(bert_emb.device)
            return torch.cat((top_emb, bert_emb), axis=1)

        return numpy.concatenate((top_emb, bert_emb), axis=1)


# Vector Quantisation model encoder
def get_vq_model(model_path):
    class Encoder():
      def __init__(self, topic_matrix, topic_vectors):
        self.tm = topic_matrix
        self.tv = topic_vectors
        self.vocabulary = set(topic_matrix.columns.tolist())
      
      def encode_doc(self, tokenized_doc):
        indeces = []
        selected_vectors = []
        probabilities = np.zeros(len(self.tm))
        for token in tokenized_doc:
          if token in self.vocabulary:
            index = np.argmax(self.tm.loc[:,token])
            indeces.append(index)
            selected_vectors.append(self.tv[index])
            probabilities[index] += 1
        if len(selected_vectors)<1:
          return np.mean(self.tv, axis = 0), np.ones(len(self.tm))/len(self.tm)
    
        return np.mean(selected_vectors, axis = 0), probabilities/len(indeces)
      
      def get_sentence_embedding_dimension(self):
        return self.tv.shape[1]
      
      def encode(self, sentences, return_probabilities = False, convert_to_tensor = False):
        sentence_vectors = []
        sentence_probabilities = []
        
        tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
        for sentence in tokenized_sentences:
            dv, pr = self.encode_doc(sentence)
            sentence_vectors.append(dv)
            sentence_probabilities.append(pr)
        
        if convert_to_tensor:
            sentence_vectors = torch.tensor(sentence_vectors)
        if return_probabilities:
            return sentence_vectors, sentence_probabilities
        return sentence_vectors
    
    topic_matrix = pd.read_csv(os.path.join(model_path, 'topic_matrix.csv'), index_col = False)
    topic_vectors = np.load(os.path.join(model_path, 'topic_vecs.npy'))
    
    encoder = Encoder(topic_matrix, topic_vectors)
    return encoder

def get_topic_bert_combined_model(model_path, model_name, max_length = None, labels = None):
    """
    Encode sentences with vector quantised topic model and with pre-trained sentence encoder, the two embeddings are concatenated.
    """
    
    encoder = TopBertEncoder(model_path, model_name, max_length, labels)
    return encoder
 
# Below dataset is to be used for hierarchical networks (e.g. TextSeg)

def readGloveFile(gloveFile):
    with open(gloveFile, 'r') as f:
        wordToGlove = {}  
        wordToIndex = {}  
        indexToWord = {}  

        for line in f:
            record = line.strip().split()
            token = record[0] 
            wordToGlove[token] = np.array(record[1:], dtype=np.float64) 
            
        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  
            wordToIndex[tok] = kerasIdx 
            indexToWord[kerasIdx] = tok 

    return wordToIndex, indexToWord, wordToGlove


def w2iGLOVE(text, wordToIndex, no_token = False, delete_punct = True):
  if no_token:
    tokens = text
  else:
    if delete_punct:
      text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
  idxs = []
  for t in tokens:
    if t not in wordToIndex:
      try:
        idxs.append(wordToIndex['UNK'])
      except KeyError:
        idxs.append(2)
    else:
      idxs.append(wordToIndex[t])
  return idxs

class WordLevelDataset(Dataset):
    def __init__(self, lines, tag_to_ix, wordToIndex, embedding_layer):
        # self.sentences = [line[0] for line in lines] # here substitute words with word2vec indeces
        self.tgt_dataset = [[tag_to_ix[tag] for tag in line[1]] for line in lines]
        self.sentence_lengths = []
        
        self.embed = embedding_layer
        
        # self.sentences = [[embedding_layer(torch.LongTensor(w2iGLOVE(sentence, wordToIndex))).detach().numpy() for sentence in line[0]] for line in lines]
        
        self.sentences = [[torch.LongTensor(w2iGLOVE(sentence, wordToIndex)).detach() for sentence in line[0]] for line in lines]
        
        for line in lines:
            # new_line = []
            self.sentence_lengths.append([len(w2iGLOVE(x, wordToIndex)) for x in line[0]])
            # pad all sentence lengths to longest in the document
            # for s in line[0]:
            #     new_line.append(s+[0 for x in range(max(self.sentence_lengths[-1])-len(s))])
            
            # # embed the sentences
            # self.sentences.append(embedding_layer(torch.LongTensor(new_line)).detach())
    
    
    def __getitem__(self, index):
        return {
            'id': torch.tensor(index),
            # 'source': torch.tensor(self.sentences[index]),
            'source': self.sentences[index],
            'target': self.tgt_dataset[index],
            'sentence_length': self.sentence_lengths[index]
        }
        
    def __len__(self):
        return len(self.sentences)
        
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        
        # sentence_lengths = [len(x) for s in samples for x in s['source']]
        sentence_lengths = [x for s in samples for x in s['sentence_length']]
        
        
        # MAIN PROBLEM: THE LENGTH 0 FROM PADDED SENTENCES IS NOT INCLUDED
        # MAYBE EASIEST AFTER ALL IS TO LOOP THROUGH SENTENCE AND THEN REPAD
        max_words = max(sentence_lengths)
        # print(max_words)
        # print(sentence_lengths)
        def merge(values, max_length = None):
            
            if len(values[0].shape)==1:
              return torch.stack(values)
            
            if max_length is None:
              max_length = max(v.size(0) for v in values)  
            #   print(max_length)
            
            if len(values[0].shape)==3:
              
              result = torch.zeros((len(values),max_length, values[0].shape[1], values[0].shape[2]))
              for i, v in enumerate(values):
                  result[i, :len(v)] = v
              return result
            
            else:
              result = torch.zeros((len(values),max_length, values[0].shape[1]))
              for i, v in enumerate(values):
                  result[i, :len(v)] = v
              return result
        
        # def merge(values, max_sentence_length):
        #     max_length = max(v.size(0) for v in values)
        #     result_out = torch.zeros((len(values),max_length, max_sentence_length, values[0].shape[2]))
        #     # print(result.shape)
            
        #     for i, v in enumerate(values):
        #         result_in = torch.zeros(len(v), max_sentence_length, values[0].shape[2])
        #         for ii, x in enumerate(v):
        #             result_in[ii, :len(x)] = x
                
        #         result_out[i, :len(v)] = result_in
                
        #     return result_out

        def merge_tags(tags, max_length = None, ignore_idx = 0):
          if max_length is None:
              max_length = max(v.size(0) for v in tags)
          result = torch.zeros((len(tags),max_length)) - ignore_idx
          for i, v in enumerate(tags):
              result[i, :len(v)] = v
          return result
          
        
        id = torch.tensor([s['id'] for s in samples])
        src_tokens = merge([merge([self.embed(x).detach() for x in s['source']], max_words) for s in samples])
        tgt_tokens = merge_tags([torch.tensor(s['target']) for s in samples], ignore_idx = 1)
        src_lengths = [len(s['source']) for s in samples]
        
        max_src_length = max(src_lengths)
        
        new_sentence_length = []
        src_index = 0
        cumulative = 0
        for index, l1 in enumerate(sentence_lengths):
            l2 = src_lengths[src_index] + cumulative
            if l2>index:
                new_sentence_length.append(l1)
            elif l2<=index:
                new_sentence_length.extend([1 for x in range(max_src_length-src_lengths[src_index])])
                cumulative += src_lengths[src_index]
                src_index += 1
                new_sentence_length.append(l1)
        
        if src_lengths[-1]<max_src_length:
            new_sentence_length.extend([1 for x in range(max_src_length-src_lengths[src_index])])
        
                
        
        sentence_lengths = torch.LongTensor(new_sentence_length)
        
        """
        0 sentence lengths are changed to 1, in order to avoid problems with LSTM class. The padded sequences will be accounted for by the
        src_lengths and empty sentences will then be represented as 0s
        """
        
        sentence_lengths[sentence_lengths==0] = 1
        
        src_sentences = [s['source'] for s in samples]
                    
        return {
                'id': id,
                'src_tokens': src_tokens,
                'src_lengths': torch.LongTensor(src_lengths),
                'tgt_tokens': tgt_tokens,
                'src_sentences': src_sentences,
                'sentence_lengths': sentence_lengths
            }

# OLD VERSION WHERE THE EMBEDDINGS ARE PRODUCED DIRECTLY IN THE MODEL
# TODO: ENABLE SWITCHING BETWEEN THESE TWO OPTIONS
# class WordLevelDataset(Dataset):
#     def __init__(self, lines, tag_to_ix, wordToIndex, no_token=False):
#         try:
#             self.unk = wordToIndex['UNK']
#         except KeyError:
#             self.unk = 2
        
#         self.sentences = [[embedding_layer(torch.LongTensor(w2iGLOVE(sentence, wordToIndex))) for sentence in line[0]] for line in lines] # CHECK THIS
#         self.tgt_dataset = [[tag_to_ix[tag] for tag in line[1]] for line in lines]
        
    
    
#     def __getitem__(self, index):
#         return {
#             'id': torch.tensor(index),
#             'source': self.sentences[index],
#             'target': self.tgt_dataset[index]
#         }
        
#     def __len__(self):
#         return len(self.sentences)
        
#     def collater(self, samples):
#         """Merge a list of samples to form a mini-batch."""
#         if len(samples) == 0:
#             return {}
#         """
#         sentence_lengths = number of words in each sentence in the batch
#         src_lengths = number of sentences in each document in the batch
#         """
#         sentence_lengths = [len(x) for s in samples for x in s['source']]
        
#         max_words = max(sentence_lengths)
        
#         def merge(values, continuous=False):
#             if len(values[0].shape)<2:
#               return torch.stack(values)
#             else:
#               max_length = max(v.size(0) for v in values)
#               result = torch.zeros((len(values),max_length, values[0].shape[1]))
#               for i, v in enumerate(values):
#                   result[i, :len(v)] = v
#               return result

#         def merge_tags(tags, max_length = None, ignore_idx = 0):
#           if max_length is None:
#               max_length = max(v.size(0) for v in tags)
#           result = torch.zeros((len(tags),max_length)) - ignore_idx
#           for i, v in enumerate(tags):
#               result[i, :len(v)] = v
#           return result
          
        
#         id = torch.tensor([s['id'] for s in samples])
#         """
#         Below I double pad the input (sentence and document level). If empty sentences are found, just a single unk token is included in the sentence.
#         """
#         # src_tokens = merge([merge_tags([torch.tensor(x) if x else torch.tensor([self.unk]) for x in s['source']], max_words) for s in samples])
#         src_tokens = merge([merge([x if x else torch.tensor([self.unk]) for x in s['source']], max_words) for s in samples])
#         tgt_tokens = merge_tags([torch.tensor(s['target']) for s in samples], ignore_idx = 1)
#         src_lengths = [len(s['source']) for s in samples]
        
#         max_src_length = max(src_lengths)
#         """
#         here below I add a placeholder length of 0 for the padded sentences, this is needed for the word-level BiLSTM to work.
        
#         sentence_length.shape = (number of real sentences in the batch)
        
#         new_sentence_length.shape = (number of sentences in the batch including padding)
        
#         thanks to src_lengths variable the padded sentences will be brought back to zero by masking them directly in the model.
#         """
#         new_sentence_length = []
#         src_index = 0
#         cumulative = 0
#         for index, l1 in enumerate(sentence_lengths):
#             l2 = src_lengths[src_index] + cumulative
#             if l2>index:
#                 new_sentence_length.append(l1)
#             elif l2<=index:
#                 new_sentence_length.extend([0 for x in range(max_src_length-src_lengths[src_index])])
#                 cumulative += src_lengths[src_index]
#                 src_index += 1
#                 new_sentence_length.append(l1)
        
#         if src_lengths[-1]<max_src_length:
#             new_sentence_length.extend([1 for x in range(max_src_length-src_lengths[src_index])])
        
#         sentence_lengths = torch.LongTensor(new_sentence_length)
        
#         """
#         0 sentence lengths are changed to 1, in order to avoid problems with LSTM class. The padded sequences will be accounted for by the
#         src_lengths and empty sentences will then be represented as 0s
#         """
#         sentence_lengths[sentence_lengths==0] = 1
        
#         src_sentences = [s['source'] for s in samples]
                    
#         return {
#                 'id': id,
#                 'src_tokens': src_tokens.type(torch.long),
#                 'src_lengths': torch.LongTensor(src_lengths),
#                 'tgt_tokens': tgt_tokens,
#                 'src_sentences': src_sentences,
#                 'sentence_lengths': sentence_lengths
#             }

# InferSent Model below is taken from https://github.com/facebookresearch/InferSent
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This file contains the definition of encoders used in https://arxiv.org/pdf/1705.02364.pdf
"""


class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx:stidx + bsize])
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

    def visualize(self, sent, tokenize=True):

        sent = sent.split() if not tokenize else self.tokenize(sent)
        sent = [[self.bos] + [word for word in sent if word in self.word_vec] + [self.eos]]

        if ' '.join(sent[0]) == '%s %s' % (self.bos, self.eos):
            import warnings
            warnings.warn('No words in "%s" have w2v vectors. Replacing \
                           by "%s %s"..' % (sent, self.bos, self.eos))
        batch = self.get_batch(sent)

        if self.is_cuda():
            batch = batch.cuda()
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        # output, idxs = output.squeeze(), idxs.squeeze()
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

        # visualize model
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0 * n / np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()

        return output, idxs

    