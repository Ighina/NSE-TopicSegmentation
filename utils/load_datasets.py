import os
import sys
import json
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

from utils.wiki_loader_sentences import *
from utils.choiloader_sentences import *

def expand_label(labels,sentences):
  new_labels = [0 for i in range(len(sentences))]
  for i in labels:
    try:
        new_labels[i] = 1
    except IndexError:
        pass
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

def load_dataset(dataset, 
                 delete_last_sentence = False, 
                 compute_confidence_intervals = False,
                 use_end_boundary = False,
                 mask_inner_sentences = False,
                 mask_probability = 0.7,
                skip_preface = False):
    
    """
    Load all the available datasets. The function can be expanded, provided that in each case the output should be in the form of a list of tuples.
    where each tuple contains a fold of the processed dataset (just one fold/tuple if not using cross-validation).
    
    Note:
    Mask_inner_sentences is deprecated as dropping negative sentences all together disrupt the entire sequence.
    """
    
    np.random.seed(1)

    if dataset == 'BBC':
        with open('data/BBC/train.json', encoding="utf-8") as f:
            train = json.load(f)
            
        train_d = []
        
        for show in train['Transcripts']:
            sents_list = []
            labs = []
            
            for segment in show['Items']:
                sentences = nltk.sent_tokenize(segment)
                popped = 0
                if mask_inner_sentences:
                    for i in range(1,len(sentences)-1):
                        
                        if np.random.rand()>mask_probability:
                            
                            popped+=1
                            
                            sentences.pop(i-popped)
                        
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            if not use_end_boundary:
                labs = labs[:-1]
            train_d.append([sents_list, labs, show['Date']])
            
        train_data = []
        
        for w in train_d:
          if w[0]:
            train_data.append((w[0], expand_label(w[1], w[0])))
            
        with open('data/BBC/test.json', encoding="utf-8") as f:
            test = json.load(f)
            
        test_d = []
        
        for show in test['Transcripts']:
            sents_list = []
            labs = []
            
            for segment in show['Items']:
                sentences = nltk.sent_tokenize(segment)
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            labs = labs[:-1]
            test_d.append([sents_list, labs, show['Date']])
        
        test_data = []
        
        for w in test_d:
          if w[0]:
            test_data.append((w[0], expand_label(w[1], w[0])))
        
        if compute_confidence_intervals:
            folds = cross_validation_split(test_data, 10)
        else:
            folds = [(train_data, test_data)]
    
    elif dataset == 'wikisection_en_docs':
        """
        Same as wikisection_en but here we apply all the pre-processing from Xing et al., so to compare results in a fair setting.
        """
        train_data = WikipediaDataSet('data/wikisection_en_docs_new/train')
        train_data_list = []
        
        
        for w in train_data:
            
            if w[0]:
                if delete_last_sentence and len(w[0])>len(w[1]):
                    new_labs = []
                    new_w0 = []
                    for index, sent in enumerate(w[0][:-1]):
                      if index not in w[1]:
                        new_w0.append(sent)
                      else:
                        new_labs.append(len(new_w0)-1)
                    new_labs.append(len(new_w0)-1)
                    new_labs = new_labs[:-1]
                    train_data_list.append((new_w0, expand_label(new_labs, new_w0)))
                elif mask_inner_sentences:
                    new_labs = []
                    new_w0 = []
                    for index, sent in enumerate(w[0]):
                        if np.random.rand()>mask_probability and index not in w[1]:
                            new_w0.append(sent)
                        elif index in w[1]:
                            new_w0.append(sent)
                            new_labs.append(len(new_w0)-1)
                    new_labs.append(len(new_w0)-1)
                    train_data_list.append((new_w0, expand_label(new_labs, new_w0)))
                else:
                    # if not use_end_boundary:
                    train_data_list.append((w[0], expand_label(w[1], w[0])))
                    #else:
                    #    data_list.append((w[0], expand_label(w[1], w[0])))
        
        dev_data = WikipediaDataSet('data/wikisection_en_docs_new/dev')
        dev_data_list = []
        for w in dev_data:
            
            if w[0]:
                if delete_last_sentence and len(w[0])>len(w[1]):
                    new_labs = []
                    new_w0 = []
                    for index, sent in enumerate(w[0][:-1]):
                      if index not in w[1]:
                        new_w0.append(sent)
                      else:
                        new_labs.append(len(new_w0)-1)
                    new_labs.append(len(new_w0)-1)
                    new_labs = new_labs[:-1]
                    dev_data_list.append((new_w0, expand_label(new_labs, new_w0)))
                else:
                    # if not use_end_boundary:
                    dev_data_list.append((w[0], expand_label(w[1], w[0])))
                    #else:
                    #    data_list.append((w[0], expand_label(w[1], w[0])))
                    
        test_data = WikipediaDataSet('data/wikisection_en_docs_new/test')
        test_data_list = []
        for w in test_data:
            
            if w[0]:
                if delete_last_sentence and len(w[0])>len(w[1]):
                    new_labs = []
                    new_w0 = []
                    for index, sent in enumerate(w[0][:-1]):
                      if index not in w[1]:
                        new_w0.append(sent)
                      else:
                        new_labs.append(len(new_w0)-1)
                    new_labs.append(len(new_w0)-1)
                    new_labs = new_labs[:-1]
                    test_data_list.append((new_w0, expand_label(new_labs, new_w0)))
                else:
                    # if not use_end_boundary:
                    test_data_list.append((w[0], expand_label(w[1], w[0])))
                    #else:
                    #    data_list.append((w[0], expand_label(w[1], w[0])))
        
        if compute_confidence_intervals:
            folds = cross_validation_split(test_data_list, 10)
        else:
            folds = [(train_data_list, test_data_list, dev_data_list)]
    
    elif dataset == 'wikisection_en':
        with open('data/wikisection/wikisection_en_city_train.json', encoding = 'utf-8') as f:
            train = json.load(f)
            
        with open('data/wikisection/wikisection_en_disease_train.json', encoding = 'utf-8') as f:
            train += json.load(f)
            
        train_d = []
        
        for article in train:
            sents_list = []
            labs = []
            topics = []
            idx = 0
            for segment in article['annotations']:
                # if idx==0 and skip_preface:
                #     idx+=1
                #     continue
                begin = segment["begin"]
                end = begin + segment["length"]
                topics.append(segment["sectionLabel"])
                sentences = nltk.sent_tokenize(article["text"][begin:end])
                sentences = [s for s in sentences if len(s) > 0 and s != "\n" and s!= "***LIST***."]
                if not sentences:
                    continue
                popped = 0
                if mask_inner_sentences:
                    for i in range(1,len(sentences)-1):
                        
                        if np.random.rand()>mask_probability:
                            
                            popped+=1
                            
                            sentences.pop(i-popped)
                
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            max_sents = 150 if mask_inner_sentences else 10000
            if sents_list:
                train_d.append([sents_list[:max_sents], labs, topics])
            
        train_data = []
        
        for w in train_d:
          if w[0]:
            train_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        del(train, train_d) # clearing up space
            
        with open('data/wikisection/wikisection_en_city_validation.json', encoding = 'utf-8') as f:
            valid = json.load(f)
            
        with open('data/wikisection/wikisection_en_disease_validation.json', encoding = 'utf-8') as f:
            valid += json.load(f)
            
        valid_d = []
        
        for article in valid:
            sents_list = []
            labs = []
            topics = []
            idx = 0
            for segment in article['annotations']:
                if idx==0 and skip_preface:
                    idx+=1
                    continue
                begin = segment["begin"]
                end = begin + segment["length"]
                topics.append(segment["sectionLabel"])
                sentences = nltk.sent_tokenize(article["text"][begin:end])
                sentences = [s for s in sentences if len(s) > 0 and s != "\n" and s!= "***LIST***."]
                if not sentences:
                    continue
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            max_sents = 150 if mask_inner_sentences else 10000
            if sents_list:
                valid_d.append([sents_list[:max_sents], labs, topics])
            
        valid_data = []
        
        for w in valid_d:
          if w[0]:
            valid_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        del(valid, valid_d)
        
        with open('data/wikisection/wikisection_en_city_test.json', encoding = 'utf-8') as f:
            test = json.load(f)
            
        with open('data/wikisection/wikisection_en_disease_test.json', encoding = 'utf-8') as f:
            test += json.load(f)
            
        test_d = []
        
        for article in test:
            sents_list = []
            labs = []
            topics = []
            idx = 0
            for segment in article['annotations']:
                if idx==0 and skip_preface:
                    idx+=1
                    continue
                begin = segment["begin"]
                end = begin + segment["length"]
                topics.append(segment["sectionLabel"])
                sentences = nltk.sent_tokenize(article["text"][begin:end])
                sentences = [s for s in sentences if len(s) > 0 and s != "\n" and s!= "***LIST***."]
                if not sentences:
                    continue
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            max_sents = 150 if mask_inner_sentences else 10000
            if sents_list:
                test_d.append([sents_list[:max_sents], labs, topics])
            
        test_data = []
        
        for w in test_d:
          if w[0]:
            test_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        del(test, test_d)
        
        if compute_confidence_intervals:
            folds = cross_validation_split(test_data, 10)
        else:
            folds = [(train_data, test_data, valid_data)]
    
    elif dataset == 'wikisection_en_city':
        with open('data/wikisection/wikisection_en_city_train.json', encoding = 'utf-8') as f:
            train = json.load(f)
            
        train_d = []
        
        for article in train:
            sents_list = []
            labs = []
            topics = []
            idx = 0
            for segment in article['annotations']:
                if idx==0 and skip_preface:
                    idx+=1
                    continue
                begin = segment["begin"]
                end = begin + segment["length"]
                topics.append(segment["sectionLabel"])
                sentences = nltk.sent_tokenize(article["text"][begin:end])
                sentences = [s for s in sentences if len(s) > 0 and s != "\n" and s!= "***LIST***."]
                if not sentences:
                    continue
                popped = 0
                # if mask_inner_sentences:
                #     for i in range(1,len(sentences)-1):
                        
                #         if np.random.rand()>mask_probability:
                            
                #             popped+=1
                            
                #             sentences.pop(i-popped)
                
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            max_sents = 150 if mask_inner_sentences else 10000
            #max_sents = 150
            if 0<len(sents_list)<max_sents:
                train_d.append([sents_list[:max_sents], labs, topics])
            
        train_data = []
        
        for w in train_d:
          if w[0]:
            train_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        del(train, train_d) # clearing up space
            
        with open('data/wikisection/wikisection_en_city_validation.json', encoding = 'utf-8') as f:
            valid = json.load(f)
            
        valid_d = []
        
        for article in valid:
            sents_list = []
            labs = []
            topics = []
            idx = 0
            for segment in article['annotations']:
                if idx==0 and skip_preface:
                    idx+=1
                    continue
                begin = segment["begin"]
                end = begin + segment["length"]
                topics.append(segment["sectionLabel"])
                sentences = nltk.sent_tokenize(article["text"][begin:end])
                sentences = [s for s in sentences if len(s) > 0 and s != "\n" and s!= "***LIST***."]
                if not sentences:
                    continue
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            max_sents = 150 if mask_inner_sentences else 10000
            #max_sents = 150
            if 0<len(sents_list)<max_sents:
                valid_d.append([sents_list[:max_sents], labs, topics])
            
        valid_data = []
        
        for w in valid_d:
          if w[0]:
            valid_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        del(valid, valid_d)
        
        with open('data/wikisection/wikisection_en_city_test.json', encoding = 'utf-8') as f:
            test = json.load(f)
            
        test_d = []
        
        for article in test:
            sents_list = []
            labs = []
            topics = []
            idx = 0
            for segment in article['annotations']:
                if idx==0 and skip_preface:
                    idx+=1
                    continue
                begin = segment["begin"]
                end = begin + segment["length"]
                topics.append(segment["sectionLabel"])
                sentences = nltk.sent_tokenize(article["text"][begin:end])
                sentences = [s for s in sentences if len(s) > 0 and s != "\n" and s!= "***LIST***."]
                if not sentences:
                    continue
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            max_sents = 150 if mask_inner_sentences else 10000
            #max_sents = 150
            if 0<len(sents_list)<max_sents:
                test_d.append([sents_list[:max_sents], labs, topics])
            
        test_data = []
        
        for w in test_d:
          if w[0]:
            test_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        del(test, test_d)
        
        if compute_confidence_intervals:
            folds = cross_validation_split(test_data, 10)
        else:
            folds = [(train_data, test_data, valid_data)]
    
    elif dataset == 'wikisection_en_disease':
        with open('data/wikisection/wikisection_en_disease_train.json', encoding = 'utf-8') as f:
            train = json.load(f)
            
        train_d = []
        
        for article in train:
            sents_list = []
            labs = []
            topics = []
            idx = 0
            for segment in article['annotations']:
                if idx==0 and skip_preface:
                    idx+=1
                    continue
                begin = segment["begin"]
                end = begin + segment["length"]
                topics.append(segment["sectionLabel"])
                sentences = nltk.sent_tokenize(article["text"][begin:end])
                sentences = [s for s in sentences if len(s) > 0 and s != "\n" and s!= "***LIST***."]
                if not sentences:
                    continue
                popped = 0
                if mask_inner_sentences:
                    for i in range(1,len(sentences)-1):
                        
                        if np.random.rand()>mask_probability:
                            
                            popped+=1
                            
                            sentences.pop(i-popped)
                
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            max_sents = 150 if mask_inner_sentences else 10000
            if sents_list:
                train_d.append([sents_list[:max_sents], labs, topics])
            
        train_data = []
        
        for w in train_d:
          if w[0]:
            train_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        del(train, train_d) # clearing up space
            
        with open('data/wikisection/wikisection_en_disease_validation.json', encoding = 'utf-8') as f:
            valid = json.load(f)
            
        valid_d = []
        
        for article in valid:
            sents_list = []
            labs = []
            topics = []
            idx = 0
            for segment in article['annotations']:
                if idx==0 and skip_preface:
                    idx+=1
                    continue
                begin = segment["begin"]
                end = begin + segment["length"]
                topics.append(segment["sectionLabel"])
                sentences = nltk.sent_tokenize(article["text"][begin:end])
                sentences = [s for s in sentences if len(s) > 0 and s != "\n" and s!= "***LIST***."]
                if not sentences:
                    continue
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            max_sents = 150 if mask_inner_sentences else 10000
            if sents_list:
                valid_d.append([sents_list[:max_sents], labs, topics])
            
        valid_data = []
        
        for w in valid_d:
          if w[0]:
            valid_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        del(valid, valid_d)
        
        with open('data/wikisection/wikisection_en_disease_test.json', encoding = 'utf-8') as f:
            test = json.load(f)
            
        test_d = []
        
        for article in test:
            sents_list = []
            labs = []
            topics = []
            idx = 0
            for segment in article['annotations']:
                if idx==0 and skip_preface:
                    idx+=1
                    continue
                begin = segment["begin"]
                end = begin + segment["length"]
                topics.append(segment["sectionLabel"])
                sentences = nltk.sent_tokenize(article["text"][begin:end])
                sentences = [s for s in sentences if len(s) > 0 and s != "\n" and s!= "***LIST***."]
                if not sentences:
                    continue
                if delete_last_sentence:
                    sentences = sentences[:-1]
                sents_list.extend(sentences)
                if labs:
                    labs.append(labs[-1]+len(sentences))
                else:
                    labs.append(len(sentences)-1)
            
            max_sents = 150 if mask_inner_sentences else 10000
            if sents_list:
                test_d.append([sents_list[:max_sents], labs, topics])
            
        test_data = []
        
        for w in test_d:
          if w[0]:
            test_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        del(test, test_d)
        
        if compute_confidence_intervals:
            folds = cross_validation_split(test_data, 10)
        else:
            folds = [(train_data, test_data, valid_data)]
    
    elif dataset == 'BBCAudio':
        data = []
        data_path = './data/AudioBBC/modconhack_20210604/data'
        for root, directories, files in os.walk(data_path):
            assert len(files)>0
            for file in files:
                if file[-4:] == 'json':
                    with open(os.path.join(root, file), 'rb') as f:
                        print(file)
                        
                        test = json.load(f)
                            
                        test_d = []
                        
                        sents_list = []
                        labs = []  
                        
                        cut_index = 0
                        
                        for segment in test['data']["getProgrammeById"]['segments']:
                                sentences = nltk.sent_tokenize(segment['transcript'])
                                popped = 0
                                if mask_inner_sentences:
                                    for i in range(1,len(sentences)-1):
                                        
                                        if np.random.rand()>mask_probability:
                                            
                                            popped+=1
                                            
                                            sentences.pop(i-popped)
                                
                                if delete_last_sentence:
                                    sentences = sentences[:-1]
                                sents_list.extend(sentences)
                                
                                labs.append(len(sents_list)-1)
                                
                        test_d.append([sents_list, labs, 0])
                            
                            
                        for w in test_d:
                          if w[0]:
                            data.append((w[0], expand_label(w[1], w[0])))
        
        folds = cross_validation_split(data)
        
        test = False
    
    elif dataset == 'CNN':
        data = []
        for i in range(1, 11):
          doc = read_wiki_file('data/CNN10/doc' + str(i) + '.txt', remove_preface_segment=False, 
                               high_granularity=False, return_as_sentences=True)
          
          sents = []
          labs = []
          for subs in doc[0]:
            if subs.startswith('===='):
              labs.append(index)
            else:
              
              sentences = nltk.sent_tokenize(subs)
              popped = 0
              if mask_inner_sentences:
                    for i in range(1,len(sentences)-1):
                        
                        if np.random.rand()>mask_probability:
                            
                            popped+=1
                            
                            sentences.pop(i-popped)
              
              if delete_last_sentence:
                sentences = sentences[:-1]

              sents.extend(sentences)
              index = len(sents)-1
          labs.append(len(sents)-1)
          path = 'data/CNN10/doc' + str(i) + '.txt'
          data.append([sents, labs, path])
        
        data_list = []
        
        for w in data:
          if w[0]:
            data_list.append((w[0], expand_label(w[1], w[0])))
        
        folds = cross_validation_split(data_list)
    
    elif dataset == 'wiki':
        data = WikipediaDataSet('data/wiki_test_50', folder=True, only_letters=False)
        data_list = []
        for w in data:
            
            if w[0]:
                if delete_last_sentence:
                    new_labs = []
                    new_w0 = []
                    for index, sent in enumerate(w[0][:-1]):
                      if index not in w[1]:
                        new_w0.append(sent)
                      else:
                        new_labs.append(len(new_w0)-1)
                    new_labs.append(len(new_w0)-1)
                    new_labs = new_labs[:-1]
                    data_list.append((new_w0, expand_label(new_labs, new_w0)))
                else:
                    # if not use_end_boundary:
                    data_list.append((w[0], expand_label(w[1][:-1], w[0])))
                    #else:
                    #    data_list.append((w[0], expand_label(w[1], w[0])))
        
        folds = cross_validation_split(data_list)

    elif dataset == 'icsi':
        data = []
        
        segment_dir = 'data/icsi_mrda+hs_corpus_050512/segments'
        
        segment_files = os.listdir(segment_dir)
        
        file_dir = 'data/icsi_mrda+hs_corpus_050512/data'
        
        for root, direct, files in os.walk(file_dir):
            for file in files:
                if file[-4:]=='dadb':
                    continue
                
                try:
                    seg_file = [x for x in segment_files if re.search(file[:-6], x)][0]
                    
                    seg = []
                    
                    with open(os.path.join(segment_dir, seg_file)) as f:
                        for line in f:
                            seg.append(re.findall('\d+\.\d+', line)[0])
                
                except IndexError:
                    continue
                
                df = pd.read_csv(os.path.join(root,file), header = None)
                
                tmp = pd.DataFrame(df.iloc[:,0].str.split('_').tolist(), columns = ['id', 'start', 'end'])
                
                df = pd.concat([df, tmp], axis = 1)
                
                segment_index = 0
                
                labs = []
                
                starts = tmp['start'].tolist()
                delete_indeces = []
                deleted = 0
                for index, i in enumerate(starts):
                    if segment_index < len(seg):
                        if int(i)>float(seg[segment_index])*1000:
                            
                            if segment_index > 0:
                                if delete_last_sentence:
                                    try:
                                        labs[-2] = 1
                                    except:
                                        pass
                                    labs = labs[:-1]
                                    delete_indeces.append(index-deleted)
                                    deleted += 1
                                
                                else:
                                    labs[-1] = 1
                            
                            segment_index += 1
                           
                    labs.append(0)
                
                if use_end_boundary:
                    labs[-1] = 1
                if delete_last_sentence:
                    new_list = df[1].tolist()
                    for delete_index in delete_indeces:
                        new_list.pop(delete_index)
                    data.append((new_list, labs))
                else:
                    data.append((df[1].tolist(), labs))
        
        folds = cross_validation_split(data)
    
    elif dataset.lower()=='qmsum':
        # TODO: add all the additional arguments (e.g. delete_last_sentence, etc.)
        def QMSUM_preprocessor(text):
            text = re.sub("\\{\\w+\\}", "", text)
            text = re.sub("[\\.,\\?\\!\\:]", "", text)
            return text
        
        train_d = []
        
        for root, dirs, files in os.walk("data/QMSum/data/ALL/train"):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    file = json.load(f)
                
                start = 0 # QMSUM counts the text spans as speaker turns...
                
                topic_list = file["topic_list"]
                
                """
                Sometimes the same topic appears in multiple locations in a meeting: in this
                instances QMSUM add multiple spans in the spans list under the same topic label.
                To correctly address these instances I am saving any further element in the span list into
                the "remainder_topics" list
                and before assuming that there is an unlabelled span I will check in this list when
                creating the segments.
                """
                
                remainder_topics = {}
                
                topics = []
                
                data = []
                
                labs = []
                
                for topic in topic_list:
                    ct = topic["topic"] # current topic
                    text_spans = topic["relevant_text_span"]
                    topics.append(ct)
                    if len(text_spans)>1:
                        if ct not in remainder_topics:
                            remainder_topics[ct] = set(int(t[1]) for t in text_spans[1:])
                        else:
                            remainder_topics[ct] = remainder_topics[ct].union(set(int(t[1]) for t in text_spans[1:]))
                    
                    seg = text_spans[0]
                    
                    """
                    In QMSum, just the spans corresponding to topics that were individuated by the annotators are included, while
                    if a portion of text has no specific topic, that span is not reported. Also, the spans relating to the same
                    topic do not need to be contiguous. For this, we insert an additional part in the code that takes care
                    of trailing text that has been left out or that have been  recorded previously (under a same topic previously in the topic span)
                    by checking that the current span is starting at the end of the previously recorded one. If the condition+
                    is not met, we include also a span from the previous end to the current start and, if the previous end is
                    recorded in previously observed topical spans, we include the label for that topic, else we include a
                    "other" placeholder to indicate that this additional span has no specific topic.
                    """
                    if int(seg[0])-1>start:
                        for t in remainder_topics:
                            if int(seg[0]) in remainder_topics[t]:
                                topics.append(t)
                            else:
                                topics.append("other")
                                
                        prev_text = file["meeting_transcripts"][start:int(seg[0])]
                        
                        if labs:
                            labs.append(labs[-1]+len(prev_text))
                        else:
                            labs.append(len(prev_text)-1)
                        
                        for t in prev_text: 
                            data.append(QMSUM_preprocessor(t["content"]))
                        
                    
                    text = file["meeting_transcripts"][int(seg[0]):int(seg[1])+1]
                    popped = 0
                    if mask_inner_sentences:
                        for i in range(1,len(text)-1):
                            
                            if np.random.rand()>mask_probability:
                                
                                popped+=1
                                
                                sentences.pop(i-popped)
                    if labs:
                        labs.append(labs[-1]+len(text))
                    else:
                        labs.append(len(text)-1)
                    
                    for t in text: 
                        data.append(QMSUM_preprocessor(t["content"]))
                        
                    start = int(seg[1])
                        
                train_d.append((data, labs, topics))
                
        train_data = []
                
        for w in train_d:
          if w[0]:
            train_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        valid_d = []
        
        for root, dirs, files in os.walk("data/QMSum/data/ALL/val"):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    file = json.load(f)
                
                start = 0 # QMSUM counts the text spans as speaker turns...
                
                topic_list = file["topic_list"]
                
                """
                Sometimes the same topic appears in multiple locations in a meeting: in this
                instances QMSUM add multiple spans in the spans list under the same topic label.
                To correctly address these instances I am saving any further element in the span list into
                the "remainder_topics" list
                and before assuming that there is an unlabelled span I will check in this list when
                creating the segments.
                """
                
                remainder_topics = {}
                
                topics = []
                
                data = []
                
                labs = []
                
                for topic in topic_list:
                    ct = topic["topic"] # current topic
                    text_spans = topic["relevant_text_span"]
                    topics.append(ct)
                    if len(text_spans)>1:
                        if ct not in remainder_topics:
                            remainder_topics[ct] = set(int(t[1]) for t in text_spans[1:])
                        else:
                            remainder_topics[ct] = remainder_topics[ct].union(set(int(t[1]) for t in text_spans[1:]))
                    
                    seg = text_spans[0]
                    
                    """
                    In QMSum, just the spans corresponding to topics that were individuated by the annotators are included, while
                    if a portion of text has no specific topic, that span is not reported. Also, the spans relating to the same
                    topic do not need to be contiguous. For this, we insert an additional part in the code that takes care
                    of trailing text that has been left out or that have been  recorded previously (under a same topic previously in the topic span)
                    by checking that the current span is starting at the end of the previously recorded one. If the condition+
                    is not met, we include also a span from the previous end to the current start and, if the previous end is
                    recorded in previously observed topical spans, we include the label for that topic, else we include a
                    "other" placeholder to indicate that this additional span has no specific topic.
                    """
                    if int(seg[0])-1>start:
                        for t in remainder_topics:
                            if int(seg[0]) in remainder_topics[t]:
                                topics.append(t)
                            else:
                                topics.append("other")
                                
                        prev_text = file["meeting_transcripts"][start:int(seg[0])]
                        
                        if labs:
                            labs.append(labs[-1]+len(prev_text))
                        else:
                            labs.append(len(prev_text)-1)
                        
                        for t in prev_text: 
                            data.append(QMSUM_preprocessor(t["content"]))
                        
                    
                    text = file["meeting_transcripts"][int(seg[0]):int(seg[1])+1]
                    
                    if labs:
                        labs.append(labs[-1]+len(text))
                    else:
                        labs.append(len(text)-1)
                    
                    for t in text: 
                        data.append(QMSUM_preprocessor(t["content"]))
                    
                    start = int(seg[1])
                    
                valid_d.append((data, labs, topics))
                
        valid_data = []
                
        for w in valid_d:
          if w[0]:
            valid_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        
        test_d = []
        
        for root, dirs, files in os.walk("data/QMSum/data/ALL/test"):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    file = json.load(f)
                
                start = 0 # QMSUM counts the text spans as speaker turns...
                
                topic_list = file["topic_list"]
                
                """
                Sometimes the same topic appears in multiple locations in a meeting: in this
                instances QMSUM add multiple spans in the spans list under the same topic label.
                To correctly address these instances I am saving any further element in the span list into
                the "remainder_topics" list
                and before assuming that there is an unlabelled span I will check in this list when
                creating the segments.
                """
                
                remainder_topics = {}
                
                topics = []
                
                data = []
                
                labs = []
                
                for topic in topic_list:
                    ct = topic["topic"] # current topic
                    text_spans = topic["relevant_text_span"]
                    topics.append(ct)
                    if len(text_spans)>1:
                        if ct not in remainder_topics:
                            remainder_topics[ct] = set(int(t[1]) for t in text_spans[1:])
                        else:
                            remainder_topics[ct] = remainder_topics[ct].union(set(int(t[1]) for t in text_spans[1:]))
                    
                    seg = text_spans[0]
                    
                    """
                    In QMSum, just the spans corresponding to topics that were individuated by the annotators are included, while
                    if a portion of text has no specific topic, that span is not reported. Also, the spans relating to the same
                    topic do not need to be contiguous. For this, we insert an additional part in the code that takes care
                    of trailing text that has been left out or that have been  recorded previously (under a same topic previously in the topic span)
                    by checking that the current span is starting at the end of the previously recorded one. If the condition+
                    is not met, we include also a span from the previous end to the current start and, if the previous end is
                    recorded in previously observed topical spans, we include the label for that topic, else we include a
                    "other" placeholder to indicate that this additional span has no specific topic.
                    """
                    if int(seg[0])-1>start:
                        for t in remainder_topics:
                            if int(seg[0]) in remainder_topics[t]:
                                topics.append(t)
                            else:
                                topics.append("other")
                                
                        prev_text = file["meeting_transcripts"][start:int(seg[0])]
                        
                        if labs:
                            labs.append(labs[-1]+len(prev_text))
                        else:
                            labs.append(len(prev_text)-1)
                        
                        for t in prev_text: 
                            data.append(QMSUM_preprocessor(t["content"]))
                        
                    
                    text = file["meeting_transcripts"][int(seg[0]):int(seg[1])+1]
                    
                    if labs:
                        labs.append(labs[-1]+len(text))
                    else:
                        labs.append(len(text)-1)
                    
                    for t in text: 
                        data.append(QMSUM_preprocessor(t["content"]))
                    
                    start = int(seg[1])
                    
                test_d.append((data, labs, topics))
                
        test_data = []
                
        for w in test_d:
          if w[0]:
            test_data.append((w[0], expand_label(w[1], w[0]), w[2]))
            
        if compute_confidence_intervals:
            folds = cross_validation_split(test_data, 10)
        else:
            folds = [(train_data, test_data, valid_data)]
    
    elif dataset.lower() == "choi":
        data = ChoiDataset('data/choi')
        data_list = []
        for w in data:
          if w[0]:
            if delete_last_sentence:
                new_labs = []
                new_w0 = []
                for index, sent in enumerate(w[0][:-1]):
                  if index not in w[1]:
                    new_w0.append(sent)
                  else:
                    new_labs.append(len(new_w0)-1)
                new_labs.append(len(new_w0)-1)
                data_list.append((new_w0, expand_label(new_labs, new_w0)))
            else:
                
                data_list.append((w[0], expand_label(w[1], w[0])))
        
        folds = cross_validation_split(data_list, num_folds=7, n_test_folds = 2)
    
    elif isinstance(dataset, function):
        folds = dataset()

    else:
        raise ValueError("Dataset not recognised!")
        
    return folds