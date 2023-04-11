# Define your sentence transformer model using CLS pooling
from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, InputExample, models
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader
from utils.load_datasets import load_dataset
from TripletSimilarityLoss import TripletSimilarityLoss
import numpy as np
import argparse
import torch
from models.metrics import *
from sklearn.metrics import f1_score, precision_score, recall_score

import json

def main(args):
    
    data = load_dataset(args.dataset)
    if not args.inference:
        
        train_data = []
        
        if args.loss=="CosineSimilarity":
            pos_lab = 0.5
            neg_lab = -1.0
        elif args.loss=="TripletLoss" or args.loss=="SoftmaxLoss":
            pos_lab = 1
            neg_lab = 0
    
        add_nsp = True if args.NSP else False
    
        if add_nsp:
            train_data_nsp = []
    
        for d in data[0][0]:
            segments = []
            segment = []
            for index, sent in enumerate(d[0]): # MultipleRankingLoss (i.e. CSE loss) with hard-negatives
                segment.append(sent)
                if d[1][index]:
                    segments.append(segment)
                    segment = []
            
            for seg_idx, segment in enumerate(segments):
                for in_idx, seg in enumerate(segment):
                    if add_nsp and len(segment)>2:
                        try:
                            train_data_nsp.append(InputExample(texts=[segment[in_idx], segment[in_idx+1]], label = pos_lab))
                            in_seg_sample = in_idx
                            while in_seg_sample==in_idx: # avoid picking the same sentence
                                tmp_in_seg_sample = np.random.choice(list(range(len(segment))))
                                if tmp_in_seg_sample!=in_idx+1:
                                    in_seg_sample = np.random.choice(list(range(len(segment))))
                            train_data_nsp.append(InputExample(texts=[segment[in_idx], segment[in_seg_sample]], label = neg_lab))
                            out_seg_sample = seg_idx-1
                            train_data_nsp.append(InputExample(texts=[segment[in_idx], segments[out_seg_sample][in_seg_sample]], label = neg_lab))
                        except IndexError:
                            pass
                    if in_idx>0:
                        train_data.append(InputExample(texts=[segment[in_idx], segment[in_idx-1]], label = pos_lab))
                try:
                    train_data.append(InputExample(texts=[segment[in_idx], segments[seg_idx+1][0]], label = neg_lab))
                except IndexError:
                    train_data.append(InputExample(texts=[segment[in_idx], segments[0][0]], label = neg_lab))
    
    
        second_sentences = []
        first_sentences = []
        labels = []
    
        for d in data[0][1]:
            segments = []
            segment = []
            for index, sent in enumerate(d[0]): # MultipleRankingLoss (i.e. CSE loss) with hard-negatives
                if index:
                    if d[1][index-1]:
                        labels.append(0)
                    else:
                        labels.append(1)
                    first_sentences.append(sent)
                    second_sentences.append(d[0][index-1])
                    
                    # if d[1][index]:
                        
                        # segments.append(segment)
                        # segment = []
            
            # for seg_idx, segment in enumerate(segments):
                
            #     first_sentences.extend([segment[-1]]*2)
            #     try:
            #         second_sentences.extend([segment[0], segments[seg_idx+1][0]])
            #     except IndexError:
            #         second_sentences.extend([segment[0], segments[0][0]])
                
            #     labels.extend([1,0])
    
        evaluator = BinaryClassificationEvaluator(first_sentences, second_sentences, labels, name = "Valid_Topic_Boundaries")
        model_name = args.base_model
        word_embedding_model = models.Transformer(model_name, max_seq_length=args.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode = args.pooling)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
        # DataLoader to batch your data
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
        if args.loss=="CosineSimilarity":
            train_loss = losses.CosineSimilarityLoss(model)
        elif args.loss=="TripletLoss":
            train_loss = TripletSimilarityLoss(model)
        elif args.loss=="SoftmaxLoss":
            # This is actually the same of TripletSimilarityLoss but with sentence transformers implementation. Incidentally, that is exactly the code used by MTL paper
            train_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)
    
        objectives = [(train_dataloader, train_loss)]
        if add_nsp:
            nsp_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)
            nsp_dataloader = DataLoader(train_data_nsp, batch_size=args.batch_size, shuffle=True)
            objectives.append((nsp_dataloader, nsp_loss))
        # Call the fit method
        model.fit(
            train_objectives=objectives,
            evaluator = evaluator,
            epochs=10,
            show_progress_bar=True,
            output_path = args.output_directory
        )
    else:
        assert args.loss=="SoftmaxLoss", "inference mode works only with encoders pretrained using SoftmaxLoss"
    if args.loss=="SoftmaxLoss":
        if args.inference:
            model_name = args.base_model
            word_embedding_model = models.Transformer(model_name, max_seq_length=args.max_seq_length)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode = args.pooling)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            train_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)
            train_loss.load_state_dict(torch.load(args.inference_model))
            train_loss.to("cuda")
        else:
            torch.save(train_loss.state_dict(), args.output_directory+"_local_segmentation_model.bin")

        pks = []
        wds = []
        f1s = []
        micro_f1s = []
        precisions = []
        recalls = []

        for d in data[0][1]:
            doc = []
            true_labels = []
            for index, sent in enumerate(d[0][:-1]): # Here I flip the true labels and predict the argmin, so that it is easier to compute binary f1 for topic boundaries
                if d[1][index]:
                    true_labels.append(1)
                else:
                    true_labels.append(0)
                    
                doc.append(InputExample(texts=[sent, d[0][index+1]], label = 1))
                test_dataloader = DataLoader(doc, batch_size=args.batch_size, shuffle=False, collate_fn=train_loss.model.smart_batching_collate)

                predictions = []

                for batch in test_dataloader:
                    feats,_ = batch
                    print(train_loss.model._target_device)
                    feats = list(map(lambda b: batch_to_device(b, train_loss.model._target_device), feats))
                    with torch.no_grad():
                        _, out = train_loss(feats, None)
                        predictions.extend(torch.argmin(out, axis = 1).cpu().tolist())

                
                f1s.append(f1_score(true_labels, predictions, average="binary"))
                precisions.append(precision_score(true_labels, predictions, average="binary"))
                recalls.append(recall_score(true_labels, predictions, average="binary"))
                micro_f1s.append(f1_score(true_labels, predictions, average="micro"))

                pks.append(compute_Pk(np.array(predictions), np.array(true_labels)))
                wds.append(compute_window_diff(np.array(predictions), np.array(true_labels)))

        results = {"F1": np.mean(f1s), "Micro_F1": np.mean(micro_f1s), "Precision": np.mean(precisions), "Recall": np.mean(recalls), "Pk": float(np.mean(pks)), "WD": float(np.mean(wds))}
        with open(args.output_directory+"_local_segmentation_model_resuts.json", "w") as f:
            json.dump(results, f)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Fine-tuning of pre-trained Sentence Encoders for Topic Segmentation',
                    description='Pre-train sentence encoders using topic segmentation datasets')

    parser.add_argument('--dataset', '-data', type=str, required=True, help='The datasets used for pre-training')
    parser.add_argument('--base_model', '-model', type=str, required=True, help='The pre-trained model to fine-tune. It can be any  transformer encoder models for NLP from huggingface hub. Models from sentence transformers library are recommended.')
    parser.add_argument('--output_directory', '-out', type=str, required=True, help='The directory where to store the fine-tuned model.')
    
    parser.add_argument('--loss', '-l', choices=["CosineSimilarity", "TripletLoss", "SoftmaxLoss"], default="CosineSimilarity", help='The loss to minimise in pre-training')
    parser.add_argument('--pooling', '-p', choices=["cls", "mean", "max"], help='The pooling strategy to obtain sentence embeddings.')
    parser.add_argument('--max_seq_length', '-seq', default=128, type=int, help='The maximum length allowed for input sentences.')
    parser.add_argument('--batch_size', '-bs', default=32, type=int, help='The batch size in fine-tuning.')
    parser.add_argument('--NSP', action="store_true", help='Whether to include also the next sentence prediction task in fine-tuning')
    parser.add_argument('--inference', action="store_true", help="Use the pre-trained encoder with the linear layer it was trained on directly to perform inference on test set and report results (== MTL paper). NB it works just with encoders fine-tuned with sofrmax loss.")
    parser.add_argument('--inference_model', type=str, required=False, help="If performing inference (see above), the path to the pre-trained model.")

    args = parser.parse_args()

    main(args)