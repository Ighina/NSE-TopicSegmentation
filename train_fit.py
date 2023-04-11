# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:53:24 2021

@author: Iacopo
"""

# TODO: change the code so that the optimal hyperparameters (e.g. number of layers, hidden units, etc.) are chosen on the basis of validation rather than test results

import os
import sys
import json
import nltk
nltk.download('punkt')
import shutil
import itertools
import torch
import numpy as np
import pandas as pd
import pickle
import tensorflow_hub as hub
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profiler import PyTorchProfiler
from models.EncoderDataset import *

from utils.load_datasets import *

from models.lightning_model import *

import argparse


def main(args):
    
    diff = '_diff' if args.embedding_difference else ''
    exp = args.experiment_name + diff
    
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        
        if args.wandb_key!=0:
            wandb.login(key=args.wandb_key)
        else:
            wandb.login() # enter your API key when prompted
        
        logger = WandbLogger(log_model=False, name='enc_'+args.encoder+ diff +'_opt_'+args.optimizer+'_lr_'+str(args.learning_rate)+'_bs_'+str(args.batch_size)+'_loss_'+args.loss_function, project=args.dataset+'_'+args.architecture+'_'+args.metric, dir=os.path.split(exp)[0])
    else:
        logger = None
    
    verbose = args.verbose

    assert not os.path.exists(exp), 'The name of this experiment has already be used: please change experiment name or delete all the existent results from {} folder to use this name'.format(args.experiment_name)
    
    os.makedirs(exp)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
        args.num_gpus = 0
    
    tag_to_ix = {0:0, 1:1, '<START>':2, '<STOP>':3}
    
    bs = args.batch_size
    
    test = False
    valid = False
        
    folds = load_dataset(args.dataset,
                         delete_last_sentence = args.delete_last_sentence,
                         use_end_boundary = args.use_end_boundary,
                         mask_inner_sentences = args.mask_negatives,
                         mask_probability = args.mask_percentage)
    
    
    if len(folds)==1:
        test = True
        if len(folds[0])==3:
            valid = True
        elif len(folds[0])==2:
            pass
        else:
            raise ValueError("The returned dataset contains an incorrect number of sublists. You should return either a list of two lists if having just a training and test set or return a list of three lists if having training, test and validation sets.")
    elif len(folds[0])>3 or len(folds[0])<2:
        raise ValueError("The returned dataset contains an incorrect number of sublists. You should return either a list of two lists if having just a training and test set or return a list of three lists if having training, test and validation sets.")
        

    loaders = []
    
    valid_percentage = args.valid_percentage
    second_dim = 0
    if args.architecture.lower().startswith('text'):
        encoder = None
    elif args.architecture.lower().startswith('crossencoder'):
        encoder, _ = args.encoder.split("+")
    else:
        if args.vq_bert_encoder:
            encoder = get_topic_bert_combined_model(args.vq_path, args.encoder, max_length = args.max_length, labels = args.zero_shot_labels)
        elif args.vq_encoder:
            encoder = get_vq_model(args.vq_path)
        else:
            encoder = get_model(args.encoder, max_length = args.max_length, labels = args.zero_shot_labels, emb_diff = args.embedding_difference)[0]
        if isinstance(args.second_encoder, str):
            second_encoder = get_model(args.second_encoder, max_length = args.max_length, labels = args.zero_shot_labels, emb_diff = args.embedding_difference)[0]
            second_dim = second_encoder.get_sentence_embedding_dimension()
        else:
            second_encoder = None
            
            
    
        in_dim = encoder.get_sentence_embedding_dimension()+second_dim
    
    os.chdir(args.experiment_name)
    
    precompute = not args.online_encoding
    
    CRF = True if args.architecture.lower().endswith('crf') else False
    
    if verbose and CRF:
        print('Using architecture with conditional random field layer')
    
    # WordMatrix = None # This variable is active just if using the TextSeg baseline
    
    if args.architecture.lower().startswith('text'):
        
        import gensim.downloader
        
        WordMatrix = gensim.downloader.load('word2vec-google-news-300')
        
        word2index = {w:i for i, w in enumerate(WordMatrix.index_to_key)}
        
        WordMatrix = createPreTrainedEmbedding(WordMatrix, word2index, False)
        print('Now I am here')
        for fold in folds:
            if not test: # switch to change the validation/training split
                valid_split = int(len(fold[0])*valid_percentage)
                train_dataset = WordLevelDataset(fold[0][:-valid_split], tag_to_ix, word2index, WordMatrix)
                valid_dataset = WordLevelDataset(fold[0][-valid_split:], tag_to_ix, word2index, WordMatrix)
            
            elif args.dataset.lower()=='qmsum' or args.dataset.startswith('wikisection'):
                # qmsum is the only dataset having a fixed development set
                train_dataset = WordLevelDataset(fold[0], tag_to_ix, word2index, WordMatrix)
                valid_dataset = WordLevelDataset(fold[2], tag_to_ix, word2index, WordMatrix)
            
            else:
                print('using alternate train/validation split')
                valid_num = int(valid_percentage*100)
                train_samples = []
                valid_samples = []
                for index, sample in enumerate(fold[0]):
                    if index%valid_num==0:
                        valid_samples.append(sample)
                    else:
                        train_samples.append(sample)
                
                train_dataset = WordLevelDataset(train_samples, tag_to_ix, word2index, WordMatrix)
                valid_dataset = WordLevelDataset(valid_samples, tag_to_ix, word2index, WordMatrix)
                
            test_dataset = WordLevelDataset(fold[1], tag_to_ix, word2index, WordMatrix)
            train_loader = DataLoader(train_dataset, batch_size = min(bs, len(train_dataset)), collate_fn = train_dataset.collater)
            if verbose:
                print('Train loader has: {} documents'.format(len(train_dataset)))
            valid_loader = DataLoader(valid_dataset, batch_size = min(bs, len(valid_dataset)), collate_fn=valid_dataset.collater)
            if verbose:
                print('Validation loader has: {} documents'.format(len(valid_dataset)))
            test_loader = DataLoader(test_dataset, batch_size = min(bs, len(test_dataset)), collate_fn = valid_dataset.collater)
            if verbose:
                print('Test loader has: {} documents'.format(len(test_dataset)))
            loaders.append((train_loader, valid_loader, test_loader))
    
    elif args.architecture.lower().startswith('crossencoder'):
        for fold in folds:
            if not test: # switch to change the validation/training split
                valid_split = int(len(fold[0])*valid_percentage)
                train_dataset = CrossEncoderDataset(fold[0][:-valid_split], enc=encoder, minus = CRF, longformer = False)
                valid_dataset = CrossEncoderDataset(fold[0][-valid_split:], enc=encoder,minus = CRF, longformer = False)
            
            elif args.dataset.lower()=='qmsum' or args.dataset.startswith('wikisection'):
                # qmsum is the only dataset having a fixed development set
                train_dataset = CrossEncoderDataset(fold[0], enc=encoder, minus = CRF, longformer = False)
                valid_dataset = CrossEncoderDataset(fold[2], enc=encoder, minus = CRF, longformer = False)
            
            else:
                print('using alternate train/validation split')
                valid_num = int(valid_percentage*100)
                train_samples = []
                valid_samples = []
                for index, sample in enumerate(fold[0]):
                    if index%valid_num==0:
                        valid_samples.append(sample)
                    else:
                        train_samples.append(sample)
                
                train_dataset = CrossEncoderDataset(train_samples, enc=encoder, minus = CRF, longformer = False)
                valid_dataset = CrossEncoderDataset(valid_samples, enc=encoder, minus = CRF, longformer = False)
                
            test_dataset = CrossEncoderDataset(fold[1], enc=encoder, minus = CRF, longformer = False)
            train_loader = DataLoader(train_dataset, batch_size = min(bs, len(train_dataset)), collate_fn = train_dataset.collater)
            if verbose:
                print('Train loader has: {} documents'.format(len(train_dataset)))
            valid_loader = DataLoader(valid_dataset, batch_size = min(bs, len(valid_dataset)), collate_fn=valid_dataset.collater)
            if verbose:
                print('Validation loader has: {} documents'.format(len(valid_dataset)))
            test_loader = DataLoader(test_dataset, batch_size = min(bs, len(test_dataset)), collate_fn = valid_dataset.collater)
            if verbose:
                print('Test loader has: {} documents'.format(len(test_dataset)))
            loaders.append((train_loader, valid_loader, test_loader))
    
    elif args.embeddings_directory is None:
        for fold in folds:
            if not test: # switch to change the validation/training split
                valid_split = int(len(fold[0])*valid_percentage)
                train_dataset = SentenceDataset(fold[0][:-valid_split], tag_to_ix, encoder = encoder, precompute = precompute, CRF =CRF, cosine_loss = args.cosine_loss, manual_max_length = args.max_doc_length, mask_inner_sentences = args.mask_negatives_in_sequence, mask_probability = args.mask_percentage, second_encoder = second_encoder)
                valid_dataset = SentenceDataset(fold[0][-valid_split:], tag_to_ix, encoder = encoder, precompute = precompute, CRF =CRF, manual_max_length = args.max_doc_length, second_encoder = second_encoder)
            
            elif args.dataset.lower()=='qmsum' or args.dataset.startswith('wikisection'):
                # qmsum is the only dataset having a fixed development set
                train_dataset = SentenceDataset(fold[0], tag_to_ix, encoder = encoder, precompute = precompute, CRF =CRF, cosine_loss = args.cosine_loss, manual_max_length = args.max_doc_length, mask_inner_sentences = args.mask_negatives_in_sequence, mask_probability = args.mask_percentage, second_encoder = second_encoder)
                valid_dataset = SentenceDataset(fold[2], tag_to_ix, encoder = encoder, precompute = precompute, CRF =CRF, manual_max_length = args.max_doc_length, second_encoder = second_encoder)
            
            else:
                print('using alternate train/validation split')
                valid_num = int(valid_percentage*100)
                train_samples = []
                valid_samples = []
                for index, sample in enumerate(fold[0]):
                    if index%valid_num==0:
                        valid_samples.append(sample)
                    else:
                        train_samples.append(sample)
                
                train_dataset = SentenceDataset(train_samples, tag_to_ix, encoder = encoder, precompute = precompute, CRF = CRF, cosine_loss = args.cosine_loss, manual_max_length = args.max_doc_length, mask_inner_sentences = args.mask_negatives_in_sequence, mask_probability = args.mask_percentage, second_encoder = second_encoder)
                valid_dataset = SentenceDataset(valid_samples, tag_to_ix, encoder = encoder, precompute = precompute, CRF = CRF, manual_max_length = args.max_doc_length, second_encoder = second_encoder)
                
            test_dataset = SentenceDataset(fold[1], tag_to_ix, encoder = encoder, precompute = precompute, CRF =CRF, manual_max_length = args.max_doc_length, second_encoder = second_encoder)
            train_loader = DataLoader(train_dataset, batch_size = min(bs, len(train_dataset)), collate_fn = train_dataset.collater)
            if verbose:
                print('Train loader has: {} documents'.format(len(train_dataset)))
            valid_loader = DataLoader(valid_dataset, batch_size = min(bs, len(valid_dataset)), collate_fn=valid_dataset.collater)
            if verbose:
                print('Validation loader has: {} documents'.format(len(valid_dataset)))
            test_loader = DataLoader(test_dataset, batch_size = min(bs, len(test_dataset)), collate_fn = valid_dataset.collater)
            if verbose:
                print('Test loader has: {} documents'.format(len(test_dataset)))
            loaders.append((train_loader, valid_loader, test_loader))
    else:
        # TODO: add the second encoder option in the case in which the embeddings are provided to the model from an external source
        if test:
            train_embeddings = load_embeddings(args.encoder + '_train', args.dataset, args.embeddings_directory)
            valid_embeddings = load_embeddings(args.encoder + '_valid', args.dataset, args.embeddings_directory)
            test_embeddings = load_embeddings(args.encoder + '_test', args.dataset, args.embeddings_directory)
            
            for fold in folds:
              if args.dataset.lower()=='qmsum' or args.dataset.startswith('wikisection'):
                  train_dataset = SentenceDataset(fold[0], tag_to_ix, encoder = encoder, embeddings = train_embeddings, CRF =CRF, cosine_loss = args.cosine_loss, manual_max_length = args.max_doc_length, mask_inner_sentences = args.mask_negatives_in_sequence, mask_probability = args.mask_percentage)
                  valid_dataset = SentenceDataset(fold[2], tag_to_ix, encoder = encoder, embeddings = valid_embeddings, CRF =CRF, manual_max_length = args.max_doc_length)
              else:
                  valid_split = int(len(fold[0])*valid_percentage)
                  train_dataset = SentenceDataset(fold[0][:-valid_split], tag_to_ix, encoder = encoder, embeddings = train_embeddings, CRF =CRF, cosine_loss = args.cosine_loss, manual_max_length = args.max_doc_length, mask_inner_sentences = args.mask_negatives_in_sequence, mask_probability = args.mask_percentage)
                  valid_dataset = SentenceDataset(fold[0][-valid_split:], tag_to_ix, encoder = encoder, embeddings = valid_embeddings, CRF =CRF, manual_max_length = args.max_doc_length)
              test_dataset = SentenceDataset(fold[1], tag_to_ix, encoder = encoder, embeddings = test_embeddings, CRF =CRF, manual_max_length = args.max_doc_length)
              train_loader = DataLoader(train_dataset, batch_size = min(bs, len(train_dataset)), collate_fn = train_dataset.collater)
              if verbose:
                print('Train loader has: {} documents'.format(len(train_dataset)))
              valid_loader = DataLoader(valid_dataset, batch_size = min(bs, len(valid_dataset)), collate_fn=valid_dataset.collater)
              if verbose:
                print('Validation loader has: {} documents'.format(len(valid_dataset)))
              test_loader = DataLoader(test_dataset, batch_size = min(bs, len(test_dataset)), collate_fn = valid_dataset.collater)
              if verbose:
                print('Test loader has: {} documents'.format(len(test_dataset)))
              loaders.append((train_loader, valid_loader, test_loader))
        else:
            os.chdir('../')
            embeddings = load_embeddings(args.encoder, args.dataset, args.embeddings_directory)
            os.chdir(args.experiment_name)
            if args.dataset == 'choi':
                embeddings = cross_validation_split(embeddings, num_folds=7, n_test_folds = 2)
            else:
                embeddings = cross_validation_split(embeddings)
            
            for index, fold in enumerate(folds):
                
              valid_split = int(len(fold[0])*valid_percentage)
              train_dataset = SentenceDataset(fold[0][:-valid_split], tag_to_ix, encoder = encoder, embeddings = embeddings[index][0][:-valid_split], CRF =CRF, cosine_loss = args.cosine_loss, manual_max_length = args.max_doc_length, mask_inner_sentences = args.mask_negatives_in_sequence, mask_probability = args.mask_percentage)
              valid_dataset = SentenceDataset(fold[0][-valid_split:], tag_to_ix, encoder = encoder, embeddings = embeddings[index][0][-valid_split:], CRF =CRF, manual_max_length = args.max_doc_length)
              test_dataset = SentenceDataset(fold[1], tag_to_ix, encoder = encoder, embeddings = embeddings[index][1], CRF =CRF, manual_max_length = args.max_doc_length)
              train_loader = DataLoader(train_dataset, batch_size = min(bs, len(train_dataset)), collate_fn = train_dataset.collater)
              if verbose:
                print('Train loader has: {} documents'.format(len(train_dataset)))
              valid_loader = DataLoader(valid_dataset, batch_size = min(bs, len(valid_dataset)), collate_fn=valid_dataset.collater)
              if verbose:
                print('Validation loader has: {} documents'.format(len(valid_dataset)))
              test_loader = DataLoader(test_dataset, batch_size = min(bs, len(test_dataset)), collate_fn = valid_dataset.collater)
              if verbose:
                print('Test loader has: {} documents'.format(len(test_dataset)))
              loaders.append((train_loader, valid_loader, test_loader))
            
    
    search_space = {'hidden_units': [args.hidden_units], 'number_layers': [args.num_layers]}
    
    best_results_test = {'F1':0, 'Pk':1, 'WD':1}
    best_results = 2 if args.metric=='Pk' or args.metric=='WD' else -1
    if args.metric=='B':
        best_results_test['B'] = 0
    
    if args.hyperparameters_search:
        hyperparameters = []
        if len(args.hidden_units_search_space)>0:
            search_space['hidden_units'] = args.hidden_units_search_space
        hyperparameters.append(search_space['hidden_units'])
        
        if len(args.number_layers_search_space)>0:
            search_space['number_layers'] = args.number_layers_search_space
        hyperparameters.append(search_space['number_layers'])
        
        if len(args.dropout_in_search_space)>0:
            search_space['dropin'] = args.dropout_in_search_space
        hyperparameters.append(search_space['dropin'])
        
        if len(args.dropout_out_search_space)>0:
            search_space['dropout'] = args.dropout_out_search_space
        hyperparameters.append(search_space['dropout'])
        
        hyperparameters = list(itertools.product(*hyperparameters))
        
        results_grid_f1 = {layer:[] for layer in search_space['number_layers']}
        results_grid_pk = {layer:[] for layer in search_space['number_layers']}
        results_grid_wd = {layer:[] for layer in search_space['number_layers']}
        
    with open('logs', 'w') as f:
        f.write('Training started all right...\n')
    
    seed_everything(args.seed, workers = True)
    
    for param_index, param_tuple in enumerate(hyperparameters):
        
        results = []
        results_valid = []
        
        hu, nl, d_in, d_out = param_tuple
        
        if args.hyperparameters_search:
            with open('logs', 'a') as f:
                f.write('Results for model with {} hidden units and {} layers...\n'.format(hu, nl))
        
        for index, segm in enumerate(loaders):
            if args.metric=='Pk' or args.metric=='WD' or not args.search_threshold:
                mode = 'min'
            else:
                mode = 'max'
            
            early_stop = EarlyStopping(
              monitor = 'valid_loss',
              patience = args.patience,
              strict = False,
              verbose = True,
              mode = mode)
            
            check_dir = 'checkpoints'
            
            if args.save_all_checkpoints:
                check_dir = check_dir + '_{}'.format(index)
            
            if not os.path.exists(check_dir):
                os.makedirs(check_dir)
            elif os.listdir(check_dir)[-1]!='best_model':
                
                os.remove(os.path.join(check_dir, os.listdir(check_dir)[-1]))
            
            
            checkpoint_callback = ModelCheckpoint(
                monitor='valid_loss',
                dirpath= check_dir,
                filename='checkpoint-{epoch:02d}-{valid_loss:.2f}-{threshold:.2f}',
                save_top_k=1,
                mode=mode,
            )
            
            if args.architecture.lower().startswith('text'):
                embed_size = WordMatrix.weight.shape[1]
            elif args.architecture.lower().startswith('crossencoder'):
                embed_size = args.encoder
            else:
                embed_size = in_dim
                
            WordMatrix = None
            
            tagset_size = 2
            
            train_loader, valid_loader, test_loader = segm
            model = TextSegmenter(architecture = args.architecture,
                                  tagset_size = tagset_size, 
                                  embedding_dim = embed_size, 
                                  hidden_dim = hu, 
                                  lr = args.learning_rate, 
                                  num_layers = nl,
                                  LSTM = args.NoLSTM,
                                  bidirectional = args.unidirectional,
                                  loss_fn = args.loss_function,
                                  dropout_in = d_in,
                                  dropout_out = d_out,
                                  batch_first = args.batch_second,
                                  optimizer = args.optimizer,
                                  positional_encoding = args.positional_encoding,
                                  nheads = args.number_heads,
                                  search_threshold = args.search_threshold,
                                  metric = args.metric,
                                  end_boundary = args.use_end_boundary,
                                  cosine_loss = args.cosine_loss,
                                  restricted = args.restricted,
                                  window_size = args.window_size,
                                  pretrained_embedding_layer = WordMatrix,
                                  auxiliary_coherence_original = args.auxiliary_coherence_original,
                                  restricted_self_attention = args.restricted_attention)
            
            if device == 'cuda':
                model.cuda()
            
            if args.sixteen_bit:
                bits = 16
            else:
                bits = 32
                
            if float(args.limit_train_batches)>1.0:
                limit_train_batches = int(args.limit_train_batches)
            else:
                limit_train_batches = float(args.limit_train_batches)
                
                
            if float(args.limit_valid_batches)>1.0:
                limit_valid_batches = int(args.limit_valid_batches)
            else:
                limit_valid_batches = float(args.limit_valid_batches)
            
            profiler = PyTorchProfiler(dirpath=".", filename="perf_logs")
            trainer = Trainer(callbacks = [early_stop, checkpoint_callback], 
                              max_epochs = args.max_epochs, 
                              gpus = args.num_gpus, 
                              auto_lr_find = args.auto_lr_finder,
                              gradient_clip_val = args.gradient_clipping,
                              precision = bits,
                              detect_anomaly = True,
                              logger = logger,
                              limit_train_batches = limit_train_batches,
                              limit_val_batches = limit_valid_batches,
                              profiler = profiler)
                              
            if args.auto_lr_finder:
                trainer.tune(model, train_loader, valid_loader)
            
            trainer.fit(model, train_loader, valid_loader)
            
            threshold = args.threshold if args.threshold else float(checkpoint_callback.best_model_path.split('=')[-1][:4])
            
            valid_loss = float(checkpoint_callback.best_model_path.split('=')[-2][:4])
            results_valid.append(valid_loss)
            
            if args.save_last_epoch:
                trainer.save_checkpoint(os.path.join(check_dir,"final=0.50.ckpt"))
                checkpoint_callback.best_model_path = os.path.join(check_dir,"final=0.50.ckpt")
                threshold = 0.5
            
            model = TextSegmenter.load_from_checkpoint(
                                  checkpoint_callback.best_model_path,
                                  architecture = args.architecture,
                                  tagset_size = tagset_size, 
                                  embedding_dim = embed_size, 
                                  hidden_dim = hu, 
                                  lr = args.learning_rate, 
                                  num_layers = nl,
                                  LSTM = args.NoLSTM,
                                  bidirectional = args.unidirectional,
                                  loss_fn = args.loss_function,
                                  dropout_in = 0.0,
                                  dropout_out = 0.0,
                                  batch_first = args.batch_second,
                                  optimizer = args.optimizer,
                                  positional_encoding = args.positional_encoding,
                                  nheads = args.number_heads,
                                  threshold = threshold,
                                  metric = args.metric,
                                  end_boundary = args.use_end_boundary,
                                  restricted = args.restricted,
                                  window_size = args.window_size,
                                  pretrained_embedding_layer = WordMatrix,
                                  auxiliary_coherence_original = args.auxiliary_coherence_original,
                                  restricted_self_attention = args.restricted_attention,
                                  bootstrap_test = args.save_test_scores)
                
            model.s_th = False # make sure you don't use the threshold lookup at testing stage
                
            results.append(trainer.test(model, test_loader))
            
            if args.metric=='F1':
                f1_label = 'test_loss'
                pk_label = 'Pk_loss'
                wd_label = 'WD_loss'
            elif args.metric == 'WD':
                f1_label = 'F1_loss'
                pk_label = 'Pk_loss'
                wd_label = 'test_loss'
            elif args.metric=='B':
                f1_label = 'b_f1'
                pk_label = 'b_precision'
                wd_label = 'b_recall'
                test_label = 'test_loss'
            
            elif args.metric=='scaiano':
                f1_label = 'test_loss'
                pk_label = 'b_precision'
                wd_label = 'b_recall'
                
            else:
                f1_label = 'F1_loss'
                pk_label = 'test_loss'
                wd_label = 'WD_loss'
            
            if args.metric == 'B' or args.metric == 'scaiano':
                with open('logs', 'a') as f:
                    f.write('Results for fold number {}\n'.format(index))
                    f.write('B_precision score: {}\n'.format(results[-1][0][pk_label]))
                    f.write('B_recall score: {}\n'.format(results[-1][0][wd_label]))
                    f.write('B_F1 score: {}\n'.format(results[-1][0][f1_label]))
                    f.write('Validation loss: {}\n'.format(results_valid[-1]))
                    if args.metric=='B': 
                        f.write('B Similarity score: {}\n'.format(results[-1][0][test_label]))
            else:
                with open('logs', 'a') as f:
                    f.write('Results for fold number {}\n'.format(index))
                    f.write('PK score: {}\n'.format(results[-1][0][pk_label]))
                    f.write('WD score: {}\n'.format(results[-1][0][wd_label]))
                    f.write('F1 score: {}\n'.format(results[-1][0][f1_label]))
                    f.write('Validation loss: {}\n'.format(results_valid[-1]))
        
        
        if test:
            f1 = results[-1][0][f1_label]
            pk = results[-1][0][pk_label]
            wd = results[-1][0][wd_label]
            if args.metric=='B':
                b = results[-1][0][test_label]
                
            if args.hyperparameters_search:     
                results_grid_f1[nl].append(f1)
                results_grid_pk[nl].append(pk)
                results_grid_wd[nl].append(wd)
            
            metrics = {'F1':f1, 'Pk': pk, 'WD': wd}
            if args.metric=='B':
                metrics['B'] = b

            use_f1 = args.metric=='F1' or args.metric=='scaiano' or args.metric == 'B'

            f1_best = (use_f1 and results_valid[-1]>=best_results)

            if f1_best or (args.metric == 'Pk' and results_valid[-1]<best_results) or (args.metric == 'WD' and results_valid[-1]<best_results):    
                best_results = results_valid[-1]
                best_results_test = metrics
                
                best_hu = hu
                best_nl = nl
                
                try:
                    os.remove(os.path.join(check_dir, 'best_model'))
                except:
                    pass
                
                new_name = os.path.join(check_dir, 'best_model')
                
                os.rename(checkpoint_callback.best_model_path, new_name)
                
        else:
            Pks = [p[0][pk_label] for p in results]
        
            F1s = [p[0][f1_label] for p in results]
                
            WDs = [p[0][wd_label] for p in results]
            if args.metric=='B':
                Bs = [p[0][test_label] for p in results]
                Avg_B = np.mean(Bs)
            
            valid_losses = [p for p in results_valid]
            
            Avg_PK = np.mean(Pks)
                
            Avg_F1 = np.mean(F1s)
                
            Avg_WD = np.mean(WDs)
            
            Valid = np.mean(valid_losses)
            
            if args.hyperparameters_search:
                
                results_grid_f1[nl].append(Avg_F1)
                results_grid_pk[nl].append(Avg_PK)
                results_grid_wd[nl].append(Avg_WD)
                
            metrics = {'F1':Avg_F1, 'Pk': Avg_PK, 'WD': Avg_WD}
            if args.metric=='B':
                metrics['B'] = Avg_B
            
            use_f1 = args.metric=='F1' or args.metric=='scaiano'

            # f1_best = (use_f1 and results_valid[-1][0]>=best_results) or (args.metric == 'B' and metrics['B']>=best_results['B'])
            f1_best = (use_f1 and Valid>=best_results)

            # if f1_best or (args.metric == 'Pk' and metrics['Pk']<best_results['Pk']) or (args.metric == 'WD' and metrics['WD']<best_results['WD']):
            if f1_best or (args.metric == 'Pk' and Valid<best_results) or (args.metric == 'WD' and Valid<best_results):    
                
                best_results = Valid
                best_results_test = metrics
                
                best_hu = hu
                best_nl = nl
                
                new_name = os.path.join(check_dir, 'best_model')
                
                os.rename(checkpoint_callback.best_model_path, new_name)
                
                def bootstrap(data, samples = 10000):
                  if isinstance(data, list):
                    data = pd.DataFrame(data)
                  boot = []
                  for sample in range(samples):
                    boot.append(data.sample(len(data), replace = True).mean()[0])
                  return boot
                
                boots = bootstrap(Pks)
                
                confidence_Pks = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
                
                boots = bootstrap(F1s)
                
                confidence_F1s = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
                
                boots = bootstrap(WDs)
                
                confidence_WDs = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
                
                if args.metric=='B':
                    boots = bootstrap(Bs)
                    confidence_Bs = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
                
                boots = bootstrap(valid_losses)
                confidence_Valid = (np.percentile(boots, 97.5) - np.percentile(boots, 2.5))/2
        
    if args.save_embeddings:
        if test:
            train_dataset.save_embeddings(args.encoder + '_train', args.dataset)
            valid_dataset.save_embeddings(args.encoder + '_valid', args.dataset)
            test_dataset.save_embeddings(args.encoder + '_test', args.dataset)
        else:
            train_dataset.embeddings = train_dataset.embeddings.extend(valid_dataset.embeddings)
            train_dataset.embeddings = train_dataset.embeddings.extend(test_dataset.embeddings)
            train_dataset.save_embeddings(args.encoder, args.dataset)
        
        args.save_embeddings = False
        
    if args.metric=='B' or args.metric=='scaiano':
        label_map = {'Pk':'Precision', 'WD': 'Recall', 'F1': 'F1'}
    else:
        label_map = {'Pk':'Pk', 'WD': 'WD', 'F1': 'F1'}

    if test:

        output = ['Results for experiment {} with following parameters:'.format(args.experiment_name),
              'Sentence encoder: {}'.format(args.encoder),
              'Neural architecture: {}'.format(args.architecture),
              'Batch size: {}'.format(args.batch_size),
              'Hidden units: {}'.format(best_hu),
              'Number of layers: {}'.format(best_nl),
              'Optimizer: {}'.format(args.optimizer),
              'Mean {} obtained is {}'.format(label_map['Pk'], best_results_test['Pk']),
              'Mean F1 obtained is {}'.format(best_results_test['F1']),
              'Mean {} obtained is {}'.format(label_map['WD'], best_results_test['WD']),
              'Best Validation Loss: {}'.format(best_results)]
        
        if args.metric=='B':
            output.append('Mean Boundary Similarity obtained is {}'.format(best_results_test['B']))
        
        if args.zero_shot_labels is not None:
            output.append('Labels: ' + str(args.zero_shot_labels))
        
    else:
        output = ['Results for experiment {} with following parameters:'.format(args.experiment_name),
                  'Sentence encoder: {}'.format(args.encoder),
                  'Neural architecture: {}'.format(args.architecture),
                  'Batch size: {}'.format(args.batch_size),
                  'Hidden units: {}'.format(best_hu),
                  'Number of layers: {}'.format(best_nl),
                  'Optimizer: {}'.format(args.optimizer),
                  'Mean {} obtained is {} with a 95% confidence interval of +- {}'.format(label_map['Pk'], best_results_test['Pk'], confidence_Pks),
                  'Mean F1 obtained is {} with a 95% confidence interval of +- {}'.format(best_results_test['F1'], confidence_F1s),
                  'Mean {} obtained is {} with a 95% confidence interval of +- {}'.format(label_map['WD'], best_results_test['WD'], confidence_WDs),
                  'Best Validation Loss: {} with a 95% confidence interval of +- {}'.format(best_results, confidence_Valid)]
        if args.metric=='B':
            output.append('Mean Boundary Similarity obtained is {} with a 95% confidence interval of +- {}'.format(best_results['B'], confidence_Bs))
        if args.zero_shot_labels is not None:
            output.append('Labels: ' + str(args.zero_shot_labels))
    
    if args.wandb:
        wandb.finish()
        
    if args.write_results:
            
        with open('results.txt', 'w') as f:
            for line in output:
                f.write('\n' + line + '\n')
                
        if args.save_test_scores:
            with open('all_test_scores.pkl', "wb") as f:
                pickle.dump(model.test_scores, f)
        
        argparse_dict = vars(args)
        with open('hyperparameters.json', 'w') as f:
            json.dump(argparse_dict, f)
        
    if args.hyperparameters_search:
        f1_results = pd.DataFrame(results_grid_f1)
        pk_results = pd.DataFrame(results_grid_pk)
        wd_results = pd.DataFrame(results_grid_wd)
        
        if args.write_results:
            f1_results.to_csv('F1_fit_results.csv')
            pk_results.to_csv('Pk_fit_results.csv')
            wd_results.to_csv('WD_fit_results.csv')
        
        return output, (f1_results, pk_results, wd_results)
    
    else:
        return output

    
if __name__ == '__main__':
    
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
    parser = MyParser(
                description = 'Run training with parameters defined in the relative json file')
    
    parser.add_argument('--experiment_name', '-exp', default = 'new_experiment', type = str,
                        help = 'The name of the current experiment (the output will be saved in a folder with the same name)')
    
    parser.add_argument('--dataset', '-data', default = 'choi', type=str,
                        help = 'The dataset to use in training. Options are choi, CNN or wiki')
    
    parser.add_argument('--batch_size', '-bs', default=64, type=int,
                        help = 'the size of each mini batch during training')
    
    parser.add_argument('--learning_rate', '-lr', default = 0.01, type = float,
                        help = 'The learning rate to be used during training')
    
    parser.add_argument('--valid_percentage', '-vp', default = 0.1, type = float,
                        help = 'The percentage of training data used for validation')
    
    parser.add_argument('--encoder', '-enc', default = 'stsb-bert-base', type = str,
                        help = 'The sentence encoder to be used to encode the sentences: possible options include all accepted values from sentence_transformers library and "use" for universal sentence encoder (DAN)')
    
    parser.add_argument('--online_encoding', '-oe', action = 'store_true',
                        help = 'If included, this option makes the dataloader encode each batch on the fly, instead of precomputing and accessing directly the stored embeddings (this option is useful if the embeddings do not fit in memory)')
    
    parser.add_argument('--embeddings_directory', '-ed', default = None, type = str,
                        help = 'The directory storing the precomputed embeddings. By default no directory is included and the embeddings are computed by the script directly. Precomputing the embeddings and storing them, however, result in massive saving of training time.')
    
    parser.add_argument('--patience', '-pat', default = 20, type = int, 
                        help = 'After how many bad iterations to stop training')
    
    parser.add_argument('--architecture', '-arc', default = 'biLSTMCRF', type = str,
                        help = 'Which neural architecture to use: implemented for now are BiLSTMCRF and Transformer-CRF.')
    
    parser.add_argument('--hidden_units', '-hu', default = 25, type = int,
                        help = 'How many hidden units to use')
    
    parser.add_argument('--num_layers', '-nl', default = 1, type = int,
                        help = 'How many layers to use')
    
    parser.add_argument('--NoLSTM', action = 'store_false',
                        help = 'If included, this option tells the network to use GRU instead of LSTM')
    
    parser.add_argument('--number_heads', '-nh', default = 8, type = int,
                        help = 'number of attention heads to be used in transformer.')
    
    parser.add_argument('--positional_encoding', '-pe', action = 'store_false',
                        help = 'if included, the option avoids using positional encoding in the transformers.')
                        
    parser.add_argument('--threshold', '-th', default = 0.0, type=float,
                        help = 'threshold to be used in inference for koshorek like models.')
    
    parser.add_argument('--unidirectional', action = 'store_false',
                        help = 'If included, this option tells the network to use a unidirectional RNN instead of bidirectional (default)')
    
    parser.add_argument('--max_length', type = int, required = False,
                        help = 'Just for Bert base and Bert news, the max input size of each sentence (if not included it default to 512 word pieces).')
    
    parser.add_argument('--dropout_in', '-d_in', default = 0.0, type = float,
                        help = 'The percentage of connections to randomly drop between embedding layer and first hidden layer during training')
    
    parser.add_argument('--dropout_out', '-d_out', default = 0.0, type = float,
                        help = 'The percentage of connections to randomly drop between last hidden layer and output layer during training')
    
    parser.add_argument('--dropout_in_search_space', '-diss', nargs='*', required=False, type=float,
                        help = 'In case the hyperparameters_search option is active, pass to this argument the dropout value before inputting to the recurrent layer to be searched in the fitting process.')
    
    parser.add_argument('--dropout_out_search_space', '-doss', nargs='*', required=False, type=float,
                        help = 'In case the hyperparameters_search option is active, pass to this argument the dropout value after the recurrent layer to be searched in the fitting process.')
    
    parser.add_argument('--batch_second', action = 'store_false',
                        help = 'If included, this option tells the network that the expected input has shape (seq_length, batch_size, embedding_size)')
    
    parser.add_argument('--optimizer', '-opt', default = 'Adam', type = str,
                        help = 'What optimizer to use: currently accepted are Adam or SGD (stochastic gradient descent)')
    
    parser.add_argument('--max_epochs', '-max', default = 100, type = int,
                        help = 'Number of training iterations after which to stop training (if early stopping mechanism does not stop the training before)')
    
    parser.add_argument('--num_gpus', '-gpus', default = 1, type = int,
                        help = 'Specify the number of gpus to use')
    
    parser.add_argument('--auto_lr_finder', '-auto_lr', action = 'store_true',
                        help = 'Include to activate the pytorch lightning functionality of finding the optimal learning rate before starting training')
    
    parser.add_argument('--save_all_checkpoints', '-savec', action = 'store_true',
                        help = 'If included, this option tells the program to save all the best checkpoints for each fold of the cross-validation process (extremely expensive memory wise)')
    
    parser.add_argument('--save_embeddings', '-savee', action = 'store_true',
                        help = 'If included, this option tells the script to save the embeddings extracted for the training, validation and test corpus. By default, these embeddings will be saved under embeddings/{encoder_name}_{one of train, validation or test}/embeddings_{sentence_number}')
    
    parser.add_argument('--use_end_boundary', '-ueb', action = 'store_true', help = 'Whether if to include the final sentence as a positive target in training (it will not be included when computing the metrics, but just for having additional positive classes in training)')
        
    parser.add_argument('--verbose', '-v', action = 'store_true',
                          help = 'Print out additional information during the training process')
    
    parser.add_argument('--write_results', '-wr', action='store_false',
                        help = 'If included, the results will not be written in results file.')
    
    parser.add_argument('--hyperparameters_search', '-hs', action = 'store_true',
                        help = 'If included, it will search for the best hidden units and layers numbers by doing a grid search among the options defined below and it will output a csv with all the results of the fitting process in addition to the standard results file.')
    
    parser.add_argument('--hidden_units_search_space', '-huss', nargs='*', required=False, type=int,
                        help = 'In case the hyperparameters_search option is active, pass to this argument the hidden units values to be searched in the fitting process.')
    
    parser.add_argument('--number_layers_search_space', '-nlss', nargs='*', required=False, type=int,
                        help = 'In case the hyperparameters_search option is active, pass to this argument the number of layers to be searched in the fitting process.')
    
    parser.add_argument('--metric', default = 'Pk', type = str, choices=['Pk', 'F1', 'WD', 'B', 'scaiano'],
                        help = 'The metric to use for determining the best hyperparameters in case the hyperparameters_search attribute is active (see above). Possible choices are Pk, F1 and WD (window difference)')
    
    parser.add_argument('--delete_last_sentence', '-dls', action = 'store_true',
                        help = 'This option deletes the last sentence from each segment in the input dataset so to test the robustness of the model under this condition.')
                        
    parser.add_argument('--zero_shot_labels', '-zsl', type = str, nargs='*',
                        help = 'If using zero shot approach (ZeroTiling) then provide the labels of the topic to be inferred')
                        
    parser.add_argument('--search_threshold', '-sth', action = 'store_true', help = 'Whether to search for the optimal threshold during training')
    
    parser.add_argument('--cosine_loss', '-cos', action = 'store_true', help = "Whether to include the cosine loss of the last layers' embeddings as an additional loss.")
    
    parser.add_argument('--gradient_clipping', '-gc', default = 0.0, type = float, help = 'The value to clip the gradient to.')
    
    parser.add_argument('--restricted', '-rsa', action = 'store_true', help = 'If using transformer, this option allows for using restricted self-attention instead of global self-attention, with a window parameter specified below.')
    
    parser.add_argument('--window_size', '-ws', default = 3, type = int, help = 'The window parameter for the restricted self-attention mechanism (see above)')
    
    parser.add_argument('--sixteen_bit', '-16b', action = 'store_true', help = 'Enable training in 16 bits, making trainer faster and less memory intensive (performance might drop)')
    
    parser.add_argument('--seed', default = 42, type = int, help = 'random seed for experiment reproducibility.')
    
    parser.add_argument('--loss_function', '-loss', choices = ["CrossEntropy", "BinaryCrossEntropy", "FocalLoss"], default = "CrossEntropy", help = "The loss function to minimise during training. Available options are CrossEntropy that treat the problem as a two class problem and BinaryCrossEntropy that treats the problem as a binary classification task (in this case an additional sigmoid is applied to the output of the model)")
    
    parser.add_argument('--vq_encoder', '-vqe', action = 'store_true', help = 'Whether to use the vector quantisation model as sentence encoder.')
    
    parser.add_argument('--vq_path', '-vqp', default = 'VQVAE_BBC', type = str, help = 'path to the pre-trained vector quantisation model components')
    
    parser.add_argument('--vq_bert_encoder', '-vbe', action = 'store_true', help = 'Use both pre-trained sentence encoder and vector quantisation encoder to produce sentence embeddings. The sentence encoder used is defined by the --encoder argument, while the path to the vq model is given by the --vq_path argument. This option overrides the --vq_encoder one.')
    
    parser.add_argument('--auxiliary_coherence_original', '-aux', action = 'store_true', help = 'Add the coherence objective defined in Xing et al., 2020')
    
    parser.add_argument('--restricted_attention', '-ra', action = 'store_true', help = 'Add the restricted attention mechanism as defined in Xing et al., 2020')
    
    parser.add_argument('--wandb', action = 'store_true', help = 'Use wandb as a logger')
    
    parser.add_argument('--wandb_key', default="0", type = str, help = "If using wandb, you can directly provide your api key with your option to avoid been prompted to insert the key (good, for example, if you're submitting the script as a job on a cluster)")
    
    parser.add_argument('--mask_negatives', action = 'store_true', help = 'Mask a percentage of negatives examples to tackle the label unbalance problem. (DEPRECATED)')
    
    parser.add_argument('--mask_negatives_in_sequence', '-mask', action = 'store_true', help = 'Mask a percentage of negatives examples to tackle the label unbalance problem. Unlike the option above, by using this option the masking involve just ignoring the masked sentences in the loss function, therefore allowing their information to be modelled by the model anyway.')
    
    parser.add_argument('--mask_percentage', default = 0.7, type = float, help = 'In case of using a mask for negative examples (see above), this parameter specifies the percentage of such examples to be left out.')
    
    parser.add_argument('--embedding_difference', '-diff', action = 'store_true', help = 'Whether to use the difference between adjacent embeddings as the feature for the segmentation system.')
    
    parser.add_argument('--save_test_scores', action = 'store_true', help = 'If included, with this option the test scores are saved and written in the results folder, so that alternative forms of testing can then be employed.')
    
    parser.add_argument('--max_doc_length', required = False, type = int, help = "If included it manually gives the number of sentences to which pad/truncate documents.")
    
    parser.add_argument('--limit_train_batches', default = 1.0, help = "An option to specify how many batches per training epoch to process. This option is included to replicate the setting from Kelvin et al. 2021, where they process 100 batches per epoch.")
    
    parser.add_argument('--limit_valid_batches', default = 1.0, help = "An option to specify how many batches per validation epoch to process. This option is included to replicate the setting from Kelvin et al. 2021, where they process 100 batches per epoch.")
    
    parser.add_argument('--save_last_epoch', '-s_last', action = 'store_true', help = "If included, use the model from last training epoch as checkpoint instead of the best validation one.")
    
    parser.add_argument('--second_encoder', required = False, type = str, help = "If included, this option allows the use of a second encoder to compute embeddings and concatenate them with the ones computed from the first encoder.")
    
    args = parser.parse_args()
    
    main(args)
    