# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:00:49 2021

@author: Iacopo
"""

import torch
#import pytorch_lightning as pl
import lightning.pytorch as pl
import numpy as np
from scipy.stats import mode
from sklearn.metrics import f1_score

from .CRF import *
from .metrics import *


def expand_label(labels,sentences):
  new_labels = [0 for i in range(len(sentences))]
  for i in labels:
    new_labels[i] = 1
  return new_labels

def cross_validation_split(dataset, num_folds = 5, n_test_folds = 1):
  unit_size = len(dataset)//num_folds
  test_size = len(dataset)//num_folds * n_test_folds
  folds = []
  for i in range(num_folds):
    test_start_idx = i*unit_size
    test_end_idx = i*unit_size + test_size
    test = dataset[test_start_idx:test_end_idx]
    if i == num_folds+1-n_test_folds:
        test += dataset[:test_size//n_test_folds]
        train = dataset[test_size//n_test_folds:-test_size//n_test_folds]
    else:
        train = dataset[:test_start_idx] + dataset[test_end_idx:]
    folds.append((train, test))
  return folds

def createPreTrainedEmbedding(word2Glove, word2index, isTrainable):
  
  try:
    vocab_size = len(word2index)+1
    embed_size = next(iter(word2Glove.values())).shape[0]
  except:
    vocab_size = len(word2index)
    embed_size = 300
  matrix = torch.zeros((vocab_size, embed_size))
  for word, index in word2index.items():
    matrix[index,:] = torch.FloatTensor(word2Glove[word])

  embedding = nn.Embedding.from_pretrained(matrix)
  if not isTrainable:
    embedding.requires_grad = False
  return embedding

class TextSegmenter(pl.LightningModule):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers = 1,
                 batch_first = True, LSTM = True, bidirectional = True, architecture = 'biLSTMCRF',
                 lr = 0.01, dropout_in = 0.0, dropout_out = 0.0, optimizer = 'SGD', 
                 positional_encoding = True, nheads = 8, end_boundary = False, threshold = None,
                 search_threshold = False, metric = 'Pk', cosine_loss = False, restricted = False, window_size = None,
                 pretrained_embedding_layer = None, loss_fn = "CrossEntropy", auxiliary_coherence_original = False,
                 add_embedding = False, restricted_self_attention = False, additional_lstm = True, config = None,
                 bootstrap_test = False, use_koomri_pk = False):
        super().__init__()
        if config is not None:
            tagset_size = config["tagset_size"]
            embedding_dim = config["embedding_dim"]
            hidden_dim = config["hidden_dim"]
            num_layers = config["num_layers"]
            architecture = config["architecture"]
            lr = config["learning_rate"]
            dropout_in = config["dropout_in"]
            dropout_out = config["dropout_out"]
            optimizer = config["optimizer"]
            positional_encoding = config["positional_encoding"]
            nheads = config["number_heads"]
            threshold = config["threshold"]
            search_threshold = config["search_threshold"]
            metric = config["metric"]
            cosine_loss = config["cosine_loss"]
            restricted = config["restricted_attention"]
            window_size = config["restricted_attention_window_size"]
            loss_fn = config["loss_function"]
            auxiliary_coherence_original = config["auxiliary_coherence_loss"]
            add_embedding = config["add_embedding_final"]
            restricted_self_attention = config["use_xing_restricted_self_attention"]
            additional_lstm = config["add_lstm_after_self_attention"]
            
        self.save_hyperparameters(ignore='pretrained_embedding_layer')
        self.ce = False
        
        self.boot = False
        if bootstrap_test:
            self.test_scores = []
            self.boot = True
            
        self.arc = architecture
        self.cos = cosine_loss
        if architecture == 'biLSTMCRF':
          self.cos = False
          self.model = BiRnnCrf(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 bidirectional = bidirectional, dropout_in = dropout_in, 
                 dropout_out = dropout_out, batch_first = batch_first, LSTM = LSTM,
                 architecture = 'rnn')
        elif architecture == 'Transformer-CRF':
          self.cos = False
          self.model = TransformerCRF(tagset_size, embedding_dim, hidden_dim, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, num_layers = num_layers, positional_encoding = positional_encoding, nheads = nheads, restricted = restricted,
          window_size = window_size)
        elif architecture == 'BiLSTM':
          self.model = BiLSTM(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 bidirectional = bidirectional, dropout_in = dropout_in, 
                 dropout_out = dropout_out, batch_first = batch_first, LSTM = LSTM,
                 loss_fn = loss_fn, threshold = threshold, 
                 auxiliary_coherence_original = auxiliary_coherence_original,
                 add_embedding = add_embedding, restricted_self_attention = restricted_self_attention, 
                 additional_lstm = additional_lstm)
        elif architecture == 'Transformer':
          self.model = Transformer_segmenter(tagset_size, embedding_dim, hidden_dim, num_layers = num_layers, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, loss_fn = loss_fn, positional_encoding = positional_encoding, nheads = nheads, threshold = threshold, restricted = restricted,
          window_size = window_size)
        
        elif architecture == 'BiLSTM_Transformer':
            self.model = BiLSTM_Transformer(tagset_size, embedding_dim, hidden_dim, num_layers = num_layers, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, loss_fn = loss_fn, positional_encoding = positional_encoding, nheads = nheads, restricted = restricted,
            window_size = window_size, threshold = threshold, auxiliary_coherence_original = auxiliary_coherence_original, add_embedding = add_embedding, bidirectional = bidirectional)
        
        elif architecture == 'TextSeg':
            # assert pretrained_embedding_layer is not None
            
            self.model = TextSeg(tagset_size, embedding_dim, hidden_dim, pretrained_embedding_layer, num_layers = num_layers, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, loss_fn = 'CrossEntropy', restricted_self_attention = restricted_self_attention, 
                 additional_lstm = additional_lstm, 
                 auxiliary_coherence_original = auxiliary_coherence_original)
        
        elif architecture == 'SheikhBiLSTM':
            self.model = SheikhBiLSTM(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 dropout_in = dropout_in, dropout_attention = dropout_out, batch_first = True, loss_fn = loss_fn)
        
        elif architecture == 'SheikhTransformer':
            self.model = SheikhTransformer(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 dropout_in = dropout_in, nheads = nheads, loss_fn = loss_fn, threshold = threshold, window_size = window_size)
        
        elif architecture == 'CrossEncoder':
            
            enc, pool = embedding_dim.split("+") # This seems a more logic way of adding the pooling rather than what I did in EncoderDataset.py TODO: change EncoderDataset.py accordingly
            print(enc)
            print(pool)
            self.model = CrossEncoderModel(tagset_size, enc, pool, loss_fn = 'CrossEntropy', threshold = threshold)
            self.ce = True
            
        elif architecture == 'LinearPair':
            
            self.model = Pair_MLP(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 dropout_in=dropout_in, dropout_out = dropout_out, batch_first = True, loss_fn = loss_fn, 
                 threshold = threshold)
        
        else:
          raise ValueError("No other architectures implemented yet")
        
        self.koomri_pk = use_koomri_pk # useful for replicating results from other authors
        self.learning_rate = lr
        self.optimizer = optimizer
        self.eb = end_boundary
        self.threshold = threshold
        self.s_th = search_threshold
        self.metric = metric
        self.best_th = []
        self.losses = []
        self.targets = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        
        sentence = batch['src_tokens'] 
        target = batch['tgt_tokens']
        lengths = batch['src_lengths']
        if self.ce:
            attention_mask = batch['src_attention_mask']
            lengths = [lengths, attention_mask]
        if self.arc == 'TextSeg':
            sentence_lengths = batch['sentence_lengths']
            
            loss = self.model.loss(sentence, sentence_lengths, lengths, target)
        
        else:
            if self.cos:
                segments = batch['src_segments']
            else:
                segments = None
            
            try:
                loss = self.model.loss(sentence, lengths, target, segments = segments, device = self.device)
            except TypeError:
                loss = self.model.loss(sentence, lengths, target, device = self.device)
        
        self.log('training_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     sentence = batch['src_tokens'] 
    #     target = batch['tgt_tokens']
    #     lengths = batch['src_lengths']
        
    #     loss = self.model.loss(sentence, lengths, target)
        
    #     self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        sentence = batch['src_tokens'] 
        target = batch['tgt_tokens']
        lengths = batch['src_lengths']
        if self.ce:
            attention_mask = batch['src_attention_mask']
            lengths = [lengths, attention_mask]
            
        
        if self.s_th:
            if self.arc == 'TextSeg':
                sentence_lengths = batch['sentence_lengths']
                
                # if len(sentence_lengths) != sentence.shape[0]*sentence.shape[1]:
                #     print(sentence.shape)
                #     print(sentence_lengths)
                #     0/0
                
                scores, tags = self.model(sentence, sentence_lengths, lengths)
            else:
                scores, tags = self.model(sentence, lengths, device = self.device)
                if self.ce:
                    lengths, _ = lengths
            for index, score in enumerate(scores):
                self.losses.append(score[:lengths[index]].detach().cpu().numpy())
                self.targets.append(target[index][:lengths[index]].detach().cpu().numpy())
            
        else:
            if self.arc == 'TextSeg':
                sentence_lengths = batch['sentence_lengths']
                scores, tags = self.model.loss(sentence, sentence_lengths, lengths, target)
            else:
                loss = self.model.loss(sentence, lengths, target, device = self.device)
            #self.log_dict({'valid_loss': loss, 'threshold': 0.5})
            self.losses.append(loss.detach().cpu().numpy())
        
    def on_validation_epoch_end(self):
        if self.s_th:
            scores = self.losses
            target = self.targets
            thresholds = np.arange(0.05,1,0.05)
            
            results = []
            best_idx = 0
            if self.metric.lower()=='pk' or self.metric.lower()=='wd':
                best = 1
            else:
                best = 0
            for index, th in enumerate(thresholds):
                
                if self.koomri_pk:
                    accuracy = Accuracy()
                
                if self.metric.lower()=='b' or self.metric.lower()=='scaiano':
                    loss_precision = 0
                    loss_recall = 0
                    loss_f1 = 0
                    loss_b = 0
                else:
                    loss_PK = 0
                    loss_F1 = 0
                    loss_WD = 0
                
                
                for i, tag in enumerate(scores):
                    if tag.shape[1]==2:
                        tag = tag[:,1]>th
                    else:
                        tag = tag[:,0]>th
                    
                    if self.eb:
                        tag[-1]=0
                        target[i][-1]=0
                    
                    if self.metric.lower()=='b':
                        precision, recall, f1 , b = B_measure(tag, target[i])

                        loss_precision += precision
                        loss_recall += recall
                        loss_f1 += f1
                        loss_b += b

                    elif self.metric.lower()=='scaiano':
                        precision, recall, f1 = WinPr(tag, target[i])

                        loss_precision += precision
                        loss_recall += recall
                        loss_f1 += f1
                    
                    else:
                        if self.koomri_pk:
                            accuracy.update(np.array(tag), target[i])
                        else:
                            loss_PK += float(compute_Pk(np.array(tag), target[i]))
                        
                        loss_F1 += f1_score(target[i].astype(int), np.array(tag).astype(int),
                                            labels = [1], average = None)
                  
                        try:
                            loss_WD += float(compute_window_diff(np.array(tag), target[i]))
                        except:
                            loss_WD += float(compute_Pk(np.array(tag), target[i]))
                
                if self.koomri_pk:
                    loss_PK, _ = accuracy.calc_accuracy()
                
                if self.metric.lower()=='b' or self.metric.lower()=='scaiano':
                    results.append({'b_precision': loss_precision/len(target), 'b_recall': loss_recall/len(target), 'b_f1': loss_f1/len(target)})
                    if self.metric.lower()=='b':
                        val_loss = loss_b/len(target)
                        results[-1]['valid_loss'] = val_loss
                        if val_loss>best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th
                    else:
                        val_loss = results[-1].pop('b_f1')
                        results[-1]['valid_loss'] = val_loss
                        if val_loss>best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th

                else:
                    try:
                        results.append({'Pk_loss': loss_PK/len(target), 
                           'F1_loss': (loss_F1/len(target))[0],
                           'WD_loss': loss_WD/len(target)})
                    except:
                        results.append({'Pk_loss': loss_PK/len(target), 
                           'F1_loss': loss_F1/len(target),
                           'WD_loss': loss_WD/len(target)})
                    
                    if self.metric=='F1':
                        val_loss = results[-1].pop('F1_loss')
                        results[-1]['valid_loss'] = val_loss
                        if val_loss>best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th
                        
                    elif self.metric == 'WD':
                        val_loss = results[-1].pop('WD_loss')
                        results[-1]['valid_loss'] = val_loss
                        if val_loss<best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th
                
                    else:
                        val_loss = results[-1].pop('Pk_loss')
                        results[-1]['valid_loss'] = val_loss
                        if val_loss<best:
                            best = val_loss
                            best_idx = index
                            self.best_th = th
            try:
                if results[best_idx]['valid_loss'] == 0 or results[best_idx]['valid_loss'] == 1 and self.best_th==0.1:
                    self.best_th = 0.5
                results[best_idx]['threshold'] = self.best_th
                if results[best_idx]['threshold'] is None:
                  results[best_idx]['threshold'] = 0.4
                
            except IndexError:
                results[best_idx]['threshold'] = 0.4
            
            self.best_th = []
            self.losses = []
            self.targets = []
            
            
            self.log_dict(results[best_idx], on_epoch = True, prog_bar=True)
        
        else:
            avg_loss = np.mean(self.losses)
            self.log_dict({'valid_loss':avg_loss})
        

    def test_step(self, batch, batch_idx):
        sentence = batch['src_tokens'] 
        target = batch['tgt_tokens']
        lengths = batch['src_lengths']
        if self.ce:
            attention_mask = batch['src_attention_mask']
            lengths = [lengths, attention_mask]
        
        # if self.s_th:
        #     raise NotImplementedError()
        #     # scores, tags = self.model(sentence, lengths)
        #     # for index, score in enumerate(scores):
        #     #     self.losses.append(score[:lengths[index]].detach().cpu().numpy())
        #     #     self.targets.append(target[index][:lengths[index]].detach().cpu().numpy())
        # else:
        threshold = self.threshold if self.threshold is not None else .5
            
        self.model.th = threshold
        
        if self.arc == 'TextSeg':
            sentence_lengths = batch['sentence_lengths']
            scores, tags = self.model(sentence, sentence_lengths, lengths)
        else:    
            scores, tags = self.model(sentence, lengths, device = self.device)
            if self.ce:
                lengths, _ = lengths
        
        if self.metric.lower()=='b' or self.metric.lower()=='scaiano':
            loss_precision = 0
            loss_recall = 0
            loss_f1 = 0
            loss_b = 0
        else:
            loss_PK = 0
            loss_F1 = 0
            loss_WD = 0
        
        accuracy = Accuracy()
        for i, tag in enumerate(tags):
            
            if self.eb:
                tag[-1]=0
                target[i][-1]=0
            
            if self.metric.lower()=='b':
                precision, recall, f1 , b = B_measure(tag, target[i][:lengths[i]])

                loss_precision += precision
                loss_recall += recall
                loss_f1 += f1
                loss_b += b

            elif self.metric.lower()=='scaiano':
                precision, recall, f1 = WinPr(tag, target[i][:lengths[i]])

                loss_precision += precision
                loss_recall += recall
                loss_f1 += f1
                
            else:
                
                if self.koomri_pk:
                    accuracy.update(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy())
                else:
                    loss_PK += float(compute_Pk(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
                    
                loss_F1 += f1_score(target[i][:lengths[i]].detach().cpu().numpy().astype(int), np.array(tag).astype(int),
                                        labels = [1], average = None)
                
                try:
                    loss_WD += float(compute_window_diff(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
                except:
                    loss_WD += float(compute_Pk(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
        
        if self.koomri_pk:
            loss_PK, _ = accuracy.calc_accuracy()
        
        if self.metric.lower() == 'b' or self.metric.lower()=='scaiano':
            results = {'b_precision': loss_precision/len(target), 'b_recall': loss_recall/len(target), 'b_f1': loss_f1/len(target)}
            if self.metric.lower()=='b':
                val_loss = loss_b/len(target)
                results['test_loss'] = val_loss
            else:
                val_loss = results[-1].pop('b_f1')
                results['test_loss'] = val_loss
        else:
            try:
                results = {'Pk_loss': loss_PK/len(target), 
                    'F1_loss': (loss_F1/len(target))[0],
                    'WD_loss': loss_WD/len(target),
                    'threshold': threshold}
            except:
                results = {'Pk_loss': loss_PK/len(target), 
                    'F1_loss': loss_F1/len(target),
                    'WD_loss': loss_WD/len(target),
                    'threshold': threshold}
            
            if self.metric=='F1':
                val_loss = results.pop('F1_loss')
                results['test_loss'] = val_loss
                
            elif self.metric == 'WD':
                val_loss = results.pop('WD_loss')
                results['test_loss'] = val_loss
        
            else:
                val_loss = results.pop('Pk_loss')
                results['test_loss'] = val_loss
        
        if self.boot:
            self.test_scores.extend([scores[index, :].detach().tolist()[:length.data] for index, length in enumerate(lengths)])
    
        self.log_dict(results, on_epoch = True, prog_bar=True)
    
    # def on_test_epoch_end(self):
    #     if self.s_th:
    #         scores = self.losses
    #         target = self.targets
    #         thresholds = [.05, .1, .2, .3, .4, .5, .6]
            
    #         results = []
    #         best_idx = 0
    #         best = 0 if self.metric == 'F1' else 1
    #         for index, th in enumerate(thresholds):
                
    #             loss_PK = 0
    #             loss_F1 = 0
    #             loss_WD = 0
                
                
    #             for i, tag in enumerate(scores):
    #                 tag = tag[:,1]>th
                    
    #                 loss_PK += float(compute_Pk(np.array(tag), target[i]))
    #                 loss_F1 += f1_score(target[i].astype(int), np.array(tag).astype(int),
    #                                         labels = [1], average = None)
                  
    #                 loss_WD += float(compute_window_diff(np.array(tag), target[i]))
                
                
    #             try:
    #                 results.append({'Pk_loss': loss_PK/len(target), 
    #                       'F1_loss': (loss_F1/len(target))[0],
    #                       'WD_loss': loss_WD/len(target)})
    #             except:
    #                 results.append({'Pk_loss': loss_PK/len(target), 
    #                       'F1_loss': loss_F1/len(target),
    #                       'WD_loss': loss_WD/len(target)})
                    
    #             if self.metric=='F1':
    #                 val_loss = results[-1].pop('F1_loss')
    #                 results[-1]['test_loss'] = val_loss
    #                 if val_loss>best:
    #                     best = val_loss
    #                     best_idx = index
    #                     self.best_th = th
                        
    #             elif self.metric == 'WD':
    #                 val_loss = results[-1].pop('WD_loss')
    #                 results[-1]['test_loss'] = val_loss
    #                 if val_loss<best:
    #                     best = val_loss
    #                     best_idx = index
    #                     self.best_th = th
                
    #             else:
    #                 val_loss = results[-1].pop('Pk_loss')
    #                 results[-1]['test_loss'] = val_loss
    #                 if val_loss<best:
    #                     best = val_loss
    #                     best_idx = index
    #                     self.best_th = th
    #         try:
    #             if results[best_idx]['test_loss'] == 0 or results[best_idx]['test_loss'] == 1 and self.best_th==0.05:
    #                 self.best_th = 0.5
    #             results[best_idx]['threshold'] = self.best_th
    #             if results[best_idx]['threshold'] is None:
    #               results[best_idx]['threshold'] = 0.4
                
    #         except IndexError:
    #             results[best_idx]['threshold'] = 0.4
            
    #         self.log_dict(results[best_idx])
        
    #     else:
    #         self.log_dict()
            
    def configure_optimizers(self):
        # if self.arc=='TextSeg':
        #     para = nn.ParameterList(list(self.model.embedding.parameters())+ list(self.model.word_level.parameters())+ list(self.model.sentence_level.parameters())+ list(self.model.classification.parameters()))
            
        #     optimizer = torch.optim.Adam(para, lr=self.learning_rate)
        # else:
        if self.optimizer == 'SGD':
          optimizer = torch.optim.SGD(self.parameters(),
                                     lr=self.learning_rate, weight_decay = 1e-4, momentum = 0.9)
        else:
          optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)
                                     
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'), "monitor": "valid_loss"}
         
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        
class TextSegmenter_old(pl.LightningModule):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers = 1,
                 batch_first = True, LSTM = True, bidirectional = True, architecture = 'biLSTMCRF',
                 lr = 0.01, dropout_in = 0.0, dropout_out = 0.0, optimizer = 'SGD', 
                 positional_encoding = True, nheads = 8, end_boundary = False, threshold = None,
                 search_threshold = False, metric = 'Pk'):
        super().__init__()
        self.cos = cosine_loss
        if architecture == 'biLSTMCRF':
          self.cos = False
          self.model = BiRnnCrf(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 bidirectional = bidirectional, dropout_in = dropout_in, 
                 dropout_out = dropout_out, batch_first = batch_first, LSTM = LSTM,
                 architecture = 'rnn')
        elif architecture == 'Transformer-CRF':
          self.cos = False
          self.model = TransformerCRF(tagset_size, embedding_dim, hidden_dim, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, num_layers = num_layers, positional_encoding = positional_encoding, nheads = nheads, restricted = restricted,
          window_size = window_size)
        elif architecture == 'BiLSTM':
          self.model = BiLSTM_old(tagset_size, embedding_dim, hidden_dim, num_layers=num_layers, 
                 bidirectional = bidirectional, dropout_in = dropout_in, 
                 dropout_out = dropout_out, batch_first = batch_first, LSTM = LSTM,
                 loss_fn = 'CrossEntropy', threshold = threshold)
        elif architecture == 'Transformer':
          self.model = Transformer_segmenter(tagset_size, embedding_dim, hidden_dim, num_layers = num_layers, dropout_in = dropout_in, dropout_out = dropout_out, batch_first = batch_first, loss_fn = 'CrossEntropy', positional_encoding = positional_encoding, nheads = nheads, threshold = threshold, restricted = restricted,
          window_size = window_size)

        else:
          raise ValueError("No other architectures implemented yet")
        self.learning_rate = lr
        self.optimizer = optimizer
        self.eb = end_boundary
        self.threshold = threshold
        self.s_th = search_threshold
        self.metric = metric
        self.best_th = []
        self.losses = []
        self.targets = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sentence = batch['src_tokens'] 
        target = batch['tgt_tokens']
        lengths = batch['src_lengths']
        if self.cos:
            segments = batch['src_segments']
        else:
            segments = None
        
        self.best_th = []
        self.losses = []
        self.targets = []
        
        try:
            loss = self.model.loss(sentence, lengths, target, segments = segments, device = self.device)
        except TypeError:
            loss = self.model.loss(sentence, lengths, target, device = self.device)
        
        self.log('training_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     sentence = batch['src_tokens'] 
    #     target = batch['tgt_tokens']
    #     lengths = batch['src_lengths']
        
    #     loss = self.model.loss(sentence, lengths, target)
        
    #     self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    # def on_validation_epoch_start(self):
    #     torch.cuda.empty_cache()
    
    def validation_step(self, batch, batch_idx):
        sentence = batch['src_tokens'] 
        target = batch['tgt_tokens']
        lengths = batch['src_lengths']
        
        if self.s_th:
            scores, tags = self.model(sentence, lengths, device = self.device)
            for index, score in enumerate(scores):
                self.losses.append(score[:lengths[index]].detach().cpu().numpy())
                self.targets.append(target[index][:lengths[index]].detach().cpu().numpy())
            
        else:
            loss = self.model.loss(sentence, lengths, target, device = self.device)
            self.log_dict({'valid_loss': loss, 'threshold': 0.5})
            self.losses.append(loss.item())
        
    def on_validation_epoch_end(self):
        if self.s_th:
            scores = self.losses
            target = self.targets
            thresholds = np.arange(0.15,.9,0.05)
            
            results = []
            best_idx = 0
            best = 0 if self.metric == 'F1' else 1
            for index, th in enumerate(thresholds):
                
                loss_PK = 0
                loss_F1 = 0
                loss_WD = 0
                
                
                for i, tag in enumerate(scores):
                    tag = tag[:,1]>th
                    
                    if self.eb:
                        tag[-1]=0
                        target[i][-1]=0
                    
                    loss_PK += float(compute_Pk(np.array(tag), target[i]))
                        
                    loss_F1 += f1_score(target[i].astype(int), np.array(tag).astype(int),
                                            labels = [1], average = None)
                  
                    loss_WD += float(compute_window_diff(np.array(tag), target[i]))
                
                
                try:
                    results.append({'Pk_loss': loss_PK/len(target), 
                           'F1_loss': (loss_F1/len(target))[0],
                           'WD_loss': loss_WD/len(target)})
                except:
                    results.append({'Pk_loss': loss_PK/len(target), 
                           'F1_loss': loss_F1/len(target),
                           'WD_loss': loss_WD/len(target)})
                    
                if self.metric=='F1':
                    val_loss = results[-1].pop('F1_loss')
                    results[-1]['valid_loss'] = val_loss
                    if val_loss>=best:
                        best = val_loss
                        best_idx = index
                        self.best_th = th
                        
                elif self.metric == 'WD':
                    val_loss = results[-1].pop('WD_loss')
                    results[-1]['valid_loss'] = val_loss
                    if val_loss<=best:
                        best = val_loss
                        best_idx = index
                        self.best_th = th
                
                else:
                    val_loss = results[-1].pop('Pk_loss')
                    results[-1]['valid_loss'] = val_loss
                    if val_loss<=best:
                        best = val_loss
                        best_idx = index
                        self.best_th = th
            try:
                if results[best_idx]['valid_loss'] == 0 or results[best_idx]['valid_loss'] == 1 and self.best_th==0.1:
                    self.best_th = 0.5
                results[best_idx]['threshold'] = self.best_th
                if results[best_idx]['threshold'] is None:
                  results[best_idx]['threshold'] = 0.4
                
            except IndexError:
                results[best_idx]['threshold'] = 0.4
        
            
            self.log_dict(results[best_idx], on_epoch = True, prog_bar=True)
        
        else:
            avg_loss = np.mean(self.losses)
            self.log_dict({'valid_loss':avg_loss})
        

    def test_step(self, batch, batch_idx):
        sentence = batch['src_tokens'] 
        target = batch['tgt_tokens']
        lengths = batch['src_lengths']
        
        if self.s_th:
            raise NotImplementedError()
            # scores, tags = self.model(sentence, lengths)
            # for index, score in enumerate(scores):
            #     self.losses.append(score[:lengths[index]].detach().cpu().numpy())
            #     self.targets.append(target[index][:lengths[index]].detach().cpu().numpy())
        else:
            threshold = self.threshold if self.threshold is not None else .5
                
            self.model.th = threshold
            score, tags = self.model(sentence, lengths, device = self.device)
            
            loss_PK = 0
            loss_F1 = 0
            loss_WD = 0
            
            
            for i, tag in enumerate(tags):
                
                if self.eb:
                    tag[-1]=0
                    target[i][-1]=0
                
                loss_PK += float(compute_Pk(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
                loss_F1 += f1_score(target[i][:lengths[i]].detach().cpu().numpy().astype(int) ,np.array(tag).astype(int),
                                        labels = [1], average = None)
              
                loss_WD += float(compute_window_diff(np.array(tag), target[i][:lengths[i]].detach().cpu().numpy()))
            
            
            try:
                results = {'Pk_loss': loss_PK/len(target), 
                       'F1_loss': (loss_F1/len(target))[0],
                       'WD_loss': loss_WD/len(target),
                       'threshold': threshold}
            except:
                results = {'Pk_loss': loss_PK/len(target), 
                       'F1_loss': loss_F1/len(target),
                       'WD_loss': loss_WD/len(target),
                       'threshold': threshold}
                
            if self.metric=='F1':
                val_loss = results.pop('F1_loss')
                results['test_loss'] = val_loss
                    
            elif self.metric == 'WD':
                val_loss = results.pop('WD_loss')
                results['test_loss'] = val_loss
            
            else:
                val_loss = results.pop('Pk_loss')
                results['test_loss'] = val_loss
                
        
            self.log_dict(results, on_epoch = True, prog_bar=True)
            
    def configure_optimizers(self):
        if self.arc=='TextSeg':
            para = nn.ParameterList([list(self.model.embedding.parameters()), list(self.model.word_level.parameters()), list(self.model.sentence_level.parameters()), list(self.model.classification.parameters())])
            
            optimizer = torch.optim.Adam(para, lr=self.learning_rate)
        else:
            if self.optimizer == 'SGD':
              optimizer = torch.optim.SGD(self.parameters(),
                                         lr=self.learning_rate, weight_decay = 1e-4, momentum = 0.9)
            else:
              optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.learning_rate)
                                     
        scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'), "monitor": "valid_loss"}
         
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
