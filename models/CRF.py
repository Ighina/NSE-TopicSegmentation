# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:54:27 2021

@author: Iacopo
"""
# TODO1: Add triplet loss option for auxiliary task
# TODO2: Combine LSTM and restricted multi-head attention
# TODO3: Change restricted multi-head attention to accept two inputs (forwards and backward LSTM directions)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.focal_loss import sigmoid_focal_loss
from models.RestrictedTransformerLayer import Longformer_Local_Attention, Classic_Transformer, Causal_Transformer
from models.NeuralArchitectures import *
from collections import OrderedDict
from transformers import AutoModel


def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()

def xing_restricted_attention(batch, lengths, win_size, self_attn):
    """
    Original implementation from https://github.com/lxing532/improve_topic_seg/blob/main/max_sentence_embedding.py
    As the only modification, Instead of doing all the unpadding etc. in the main body of the network, I include all in this function for portability.
    It is probably possible to do all of this without all the costly for loops, but I leave it as an optional TODO.
    """
    
    new_out = []
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=0)
    
    for b_index, X in enumerate(batch):
        doc_length = lengths[b_index]
        stacked_contexts = []
        X_new = X[:doc_length]
        for i in range(doc_length):
            if i-win_size < 0:
                Z = X_new[0:i+win_size+1]
            else:
                Z = X_new[i-win_size:i+win_size+1]
            
            X1 = Z.unsqueeze(0)
            Y1 = Z.unsqueeze(1)
            X2 = X1.repeat(Z.shape[0],1,1)
            Y2 = Y1.repeat(1,Z.shape[0],1)
        
            output = 0
        
            Z = torch.cat([X2,Y2],-1)
            if i <= win_size:
                a = Z[i,:,0:int(Z.size()[-1]/2)] 
                a_norm = a / a.norm(dim=1)[:, None]
                b = Z[i,:,int(Z.size()[-1]/2):]
                b_norm = b / b.norm(dim=1)[:, None]
                z = torch.cat([Z[i,:],sigmoid(torch.diag(torch.mm(a_norm,b_norm.transpose(0,1)))).unsqueeze(-1)],-1)
                
                attn_weight = softmax(self_attn(z)).permute(1,0)
        
                output = attn_weight.matmul(Z[i,:,0:int(Z.size()[-1]/2)])
                
            else:
                a = Z[win_size,:,0:int(Z.size()[-1]/2)] 
                a_norm = a / a.norm(dim=1)[:, None]
                b = Z[win_size,:,int(Z.size()[-1]/2):]
                b_norm = b / b.norm(dim=1)[:, None]
                z = torch.cat([Z[win_size,:],sigmoid(torch.diag(torch.mm(a_norm,b_norm.transpose(0,1)))).unsqueeze(-1)],-1)
                attn_weight = softmax(self_attn(z)).permute(1,0)
        
                output = attn_weight.matmul(Z[win_size,:,0:int(Z.size()[-1]/2)]) # shape: (hidden, 1)
            
            stacked_contexts.append(output)
        
        padded_contexts = torch.zeros(X.shape).to(X.device)
        stacked_contexts = torch.stack(stacked_contexts, axis = 0).squeeze(1)
        padded_contexts[:doc_length] = stacked_contexts
    
        new_out.append(torch.cat((X, padded_contexts), -1))

    return torch.stack(new_out, axis = 0)

def auxiliary_coherence_function(batch, targets, lengths, projection):
    """
    Code mainly readapted from https://github.com/lxing532/improve_topic_seg/blob/main/max_sentence_embedding.py
    """
    sims_outputs = []
    new_targets = []
    _, _, hidden = batch.shape
    hidden = hidden//2
    window = 1
    for i, doc_len in enumerate(lengths):
        #doc_outputs.append(padded_x[0:doc_len - 1, i, :])  # *** -1 to remove last prediction ***
        batch_x = batch[i, 0:doc_len, :]
        new_targets.append((targets[i, :doc_len-1]<1)+0.0)
        
        forward_padded_x = projection(batch_x[:-1, :hidden] - F.pad(batch_x[:-1, :hidden], (0,0,1,0), 'constant', 0)[:-window,:])
        backward_padded_x = projection(batch_x[1:, hidden:] - F.pad(batch_x[1:, hidden:], (0,0,0,1), 'constant', 0)[window:,:]).permute(1,0)  
        sims_outputs.append(F.sigmoid(torch.diag(torch.mm(forward_padded_x, backward_padded_x))))
        
    return torch.cat(sims_outputs), torch.cat(new_targets)

def aggregate_embeddings(batched_embeddings, sequence_lengths, segment_indeces, device, positive = True, max_len = 10, mean = True):
    samples1 = torch.tensor([]).to(device)
    samples2 = torch.tensor([]).to(device)
    if len(batched_embeddings.shape)==1:
        batched_embeddings = batched_embeddings.unsqueeze(0)
    for batch_index, embeddings in enumerate(batched_embeddings):
        # print(batch_index)
        
        embeddings = embeddings[:sequence_lengths[batch_index]]
        emb_size = embeddings.shape[-1]
        
        # print(embeddings.shape)
        prev_seg = 0
        for segment_index, seg in enumerate(segment_indeces[batch_index]):
            if seg-prev_seg>max_len:
                prev_seg = seg - max_len
            
            if positive:
                
                if len(embeddings[prev_seg:seg])>1:
                    first_ = embeddings[prev_seg:seg][::2][:emb_size//2]
                    second_ = embeddings[prev_seg:seg][1::2][:emb_size//2]
                    
                    # if len(first_)!=len(second_):
                    #     second_ = torch.cat((second_, second_[-1].unsqueeze(0)))
                    #     assert len(first_)==len(second_)
                    # print(first_.mean(0).shape)
                    if mean:
                        samples1 = torch.cat((samples1, first_.mean(0).unsqueeze(0)))
                        samples2 = torch.cat((samples2, second_.mean(0).unsqueeze(0)))
                    else:
                        samples1 = torch.cat((samples1, first_.sum(0).unsqueeze(0)))
                        samples2 = torch.cat((samples2, second_.sum(0).unsqueeze(0)))
            else:
                
                if mean:
                    samples1 = torch.cat((samples1, embeddings[prev_seg:seg][:emb_size//2].mean(0).unsqueeze(0)))
                else:
                    samples1 = torch.cat((samples1, embeddings[prev_seg:seg][:emb_size//2].sum(0).unsqueeze(0)))
                
                try:
                    
                    next_seg = min(segment_indeces[batch_index][segment_index+1]-seg,  seg + max_len)
                    
                    second_ = embeddings[seg:next_seg][emb_size//2:]
                    
                    if mean:
                        samples2 = torch.cat((samples2, second_.mean(0).unsqueeze(0)))
                    else:
                        samples2 = torch.cat((samples2, second_.sum(0).unsqueeze(0)))
                    
                except IndexError:
                    
                    pass
                    
                    # next_seg = min(len(embeddings), seg + max_len)
                    
                    # second_ = embeddings[seg:next_seg]
                    # second_ = embeddings[segment_indeces[segment_index-2]:prev_seg]
                
                
                
            prev_seg = seg
                
    return samples1, samples2

def cosine_loss(batched_embeddings, sequence_lengths, segment_indeces, cosine_loss_class, device):
    samples1 = torch.tensor([]).to(device)
    samples2 = torch.tensor([]).to(device)
    targets = torch.tensor([])
    
    positives = aggregate_embeddings(batched_embeddings, sequence_lengths, segment_indeces, device)
    
    samples1 = torch.cat((samples1, positives[0]), axis = 0)
    samples2 = torch.cat((samples2, positives[1]), axis = 0)
    targets = torch.cat((targets, torch.tensor([1 for x in range(len(positives[0]))])))
    
    negatives = aggregate_embeddings(batched_embeddings, sequence_lengths, segment_indeces, device, positive = False)
    
    samples1 = torch.cat((samples1, negatives[0]), axis = 0)
    samples2 = torch.cat((samples2, negatives[1]), axis = 0)
    targets = torch.cat((targets, torch.tensor([-1 for x in range(len(negatives[0]))])))
    if targets.tolist():
        loss = cosine_loss_class(samples1.to(device), samples2.to(device), targets.to(device))
        
    else:
        loss = 0
    return loss

def coherence_classification(batched_embeddings, sequence_lengths, segment_indeces, coherence_loss_class, projection, device):
    samples = torch.tensor([]).to(device)
    targets = torch.tensor([])
    
    positives = aggregate_embeddings(batched_embeddings, sequence_lengths, segment_indeces, device)
    
    positives = torch.cat(positives, axis=1)
    
    samples = torch.cat((samples, positives), axis=0)
    
    targets = torch.cat((targets, torch.tensor([1 for x in range(len(positives))])))
    
    negatives = aggregate_embeddings(batched_embeddings, sequence_lengths, segment_indeces, device, positive = False)
    
    negatives = torch.cat(positives, axis=1)
    
    samples = torch.cat((samples, negatives), axis=0)
    
    samples = projection(samples)
    
    targets = torch.cat((targets, torch.tensor([0 for x in range(len(positives))])))
    
    if targets.tolist():
        loss = coherence_loss_class(samples.to(device), targets.to(device))
        
    else:
        loss = 0
    return loss

IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from features space to tag space.
    :param in_features: number of features for the input
    :param num_tag: number of tags. DO NOT include START, STOP tags, they are included internal.
    """

    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        features = self.fc(features)
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        features = self.fc(features)

        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence
        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores


class BiRnnCrf(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=1,
                 bidirectional = True, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, LSTM = True,
                 architecture = 'rnn'):
        super(BiRnnCrf, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if architecture=='rnn':
          self.model = RNN(embedding_dim, hidden_dim, num_layers, tagset_size, 
                           bidirectional, dropout_in, dropout_out, batch_first = batch_first,
                           LSTM = LSTM)
        
        self.crf = CRF(hidden_dim*2, self.tagset_size)

    def loss(self, xs, lengths, tags, device = None):
        if device is None:
            device = self.device
        masks = create_mask(xs, lengths).to(device)
        out, features = self.model(xs, lengths)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs, lenghts, device = None):
        if device is None:
            device = self.device
        # Get the emission scores from the BiLSTM
        masks = create_mask(xs, lenghts).to(device)
        out, features = self.model(xs, lenghts)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq

class BiLSTM(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=1, 
                 bidirectional = True, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, LSTM = True,
                 loss_fn = 'CrossEntropy', threshold = None, add_embedding = False, 
                 auxiliary_coherence_original = False, restricted_self_attention = False, 
                 additional_lstm = True):
        
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size if loss_fn == "CrossEntropy" else 1
        self.add_emb = add_embedding
        # TODO: ADD DEVICE PARAMETER PASSED BY PYTORCH LIGHTNING
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

        self.model = RNN(embedding_dim, hidden_dim, num_layers, tagset_size, 
                         bidirectional, dropout_in, dropout_out, batch_first = batch_first,
                         LSTM = LSTM)
        
        if add_embedding:
            self.classification = nn.Linear(hidden_dim*2 + embedding_dim, self.tagset_size)
        else:
            self.classification = nn.Linear(hidden_dim*2, self.tagset_size)
        
        self.fl = False
        if loss_fn == 'CrossEntropy':
            self.bce = False
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
        elif loss_fn == 'BinaryCrossEntropy':
            self.bce = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
        elif loss_fn == "FocalLoss":
            self.bce = True
            self.fl = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = sigmoid_focal_loss
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')
        
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.coherence_loss = nn.BCELoss()
        self.aux = False
        if auxiliary_coherence_original:
            self.aux = True
            self.aux_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.rsa = False
        if restricted_self_attention:
            self.rsa = True
            self.self_attn = nn.Linear(hidden_dim*4+1, 1)
            self.win_size = 3 # TODO: add the option of changing the window size
            self.model2 = None
            if additional_lstm:
                self.model2 = RNN(hidden_dim*4, hidden_dim, num_layers, tagset_size, 
                                 bidirectional, dropout_in, dropout_out, batch_first = batch_first,
                                 LSTM = LSTM)
        
        self.coherence_projection = nn.Sequential(
            nn.Linear(hidden_dim*2, 1),
            nn.Sigmoid())
        
        self.softmax = nn.Softmax(dim=2)
        self.th = threshold
        
    def loss(self, xs, lengths, tags, segments = None, auxiliary_task = False, use_cosine = True, aggregate_sentences = True, device = None):
        if device is None:
            device = self.device
        _, x = self.model(xs, lengths)
        
        if self.aux:
            
            sim, sim_tags = auxiliary_coherence_function(x, tags, lengths, self.aux_projection)
            
            coh_loss = self.coherence_loss(sim, sim_tags.to(sim.device))
            
            
            
            # if self.add_emb:
            #     x = self.classification(torch.cat((x, xs), axis = 1)) if aggregate_sentences else self.classification(torch.cat((x, xs), axis = 2))
            # else:
            #     x = self.classification(x)
            
            # if self.bce:
            #     x = self.sigmoid(x)
            #     loss = self.loss_fn(x.reshape(-1), tags.reshape(-1).to(x.device))
            # else:
            #     loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(x.device))
            
            # tot_loss = 0.2*coh_loss + 0.8*loss
        
            # return tot_loss
            
            
        elif segments is not None:
            # This option represent a series of experiments to substitute the auxiliary task by Carenini et al. No success so far.
            
            if use_cosine:
                coh_loss = cosine_loss(x, lengths, segments, self.cosine_loss, device)
            else:
                coh_loss = coherence_classification(x, lengths, segments, self.coherence_loss, self.coherence_projection, device)
            
            
            # if aggregate_sentences:
            #     outputs = []
            #     targets = []
            #     if self.add_emb:
            #         embs = []
                
            #     for index, length in enumerate(lengths):
            #         outputs.append(x[index][:length])
            #         targets.append(tags[index][:length])
            #         if self.add_emb:
            #             embs.append(xs[index][:length])
                
            #     x = torch.cat(outputs, axis = 0).to(device)
            #     tags = torch.cat(targets, axis = 0)
            #     if self.add_emb:
            #         xs = torch.cat(embs, axis = 0).to(device)
                    
            
            
            # if self.add_emb:
            #     x = self.classification(torch.cat((x, xs), axis = 1)) if aggregate_sentences else self.classification(torch.cat((x, xs), axis = 2))
            # else:
            #     x = self.classification(x)
            
            # if self.bce:
            #     x = self.sigmoid(x)
            #     loss = self.loss_fn(x.reshape(-1), tags.reshape(-1).to(device))
            # else:
            #     loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(device))
            
            # tot_loss = 0.2*coh_loss + 0.8*loss
        
            # return tot_loss
        
        # else:
        
        if self.rsa:
            x = xing_restricted_attention(x, lengths, self.win_size, self.self_attn)
            if self.model2 is not None:
                _, x = self.model2(x, lengths) 
        
        if aggregate_sentences:
            # if self.bce:
            #     x = self.sigmoid(x)
            
            # loss = None
            # Just an option to ignore padding completely with a brute force approach: in case of BCE loss this option might be prefer as there is no option to ignore the padded indeces in the loss (contrary to CE)
            outputs = []
            targets = []
            if self.add_emb:
                embs = []
            
            for index, length in enumerate(lengths):
                outputs.append(x[index][:length])
                targets.append(tags[index][:length])
                if self.add_emb:
                    embs.append(xs[index][:length])
                
            #     if loss is None:
                    
            #         loss = self.loss_fn(x[index][:length], tags[index][:length])
                    
            #     else:
            #         loss += self.loss_fn(x[index][:length], tags[index][:length])
                    
            # loss /= index+1
            
            # if self.aux or segment is not None:
            #     loss = coh_loss*0.2 + 0.8*loss
            
            # return loss
            
            x = torch.cat(outputs, axis = 0).to(device)
            tags = torch.cat(targets, axis = 0)
            if self.add_emb:
                xs = torch.cat(embs, axis = 0).to(device)
        
        if self.add_emb:
            x = self.classification(torch.cat((x, xs), axis = 1)) if aggregate_sentences else self.classification(torch.cat((x, xs), axis = 2))
        else:
            x = self.classification(x)
        
                    
        if self.bce:
            if not self.fl:
                x = self.sigmoid(x)
            loss = self.loss_fn(x.reshape(-1), tags.reshape(-1).to(device))
        else:
            loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(device))
        
        if self.aux or segments is not None:
            loss = coh_loss*0.2 + 0.8*loss
        
        return loss

    def forward(self, xs, lenghts, threshold = 0.4, device = None):
        if device is None:
            device = self.device
        # Get the emission scores from the BiLSTM
        _, x = self.model(xs, lenghts)
        
        if self.rsa:
            x = xing_restricted_attention(x, lenghts, self.win_size, self.self_attn)
            if self.model2 is not None:
                _, x = self.model2(x, lenghts)    
        
        if self.add_emb:
            scores = self.classification(torch.cat((x, xs), axis = 1)) if aggregate_sentences else self.classification(torch.cat((x, xs), axis = 2))
        else:
            scores = self.classification(x)
        
        if self.th is not None:
            threshold = self.th
        
        if self.bce:
            scores = self.sigmoid(scores)
            tag_seq = scores[:,:,0]>threshold
        else:
            scores = self.softmax(scores)
            tag_seq = scores[:,:,1]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]
        
class BiLSTM_old(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=1, 
                 bidirectional = True, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, LSTM = True,
                 loss_fn = 'CrossEntropy', threshold = None):
        super(BiLSTM_old, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

        self.model = RNN(embedding_dim, hidden_dim, num_layers, tagset_size, 
                         bidirectional, dropout_in, dropout_out, batch_first = batch_first,
                         LSTM = LSTM)
        
        self.classification = nn.Linear(hidden_dim*2, self.tagset_size)
        
        if loss_fn == 'CrossEntropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
        elif loss_fn == 'BinaryCrossEntropy':
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')
        
        
        self.softmax = nn.Softmax(dim=2)
        self.th = threshold
        
    def loss(self, xs, lengths, tags, segments = None, use_cosine = True, aggregate_sentences = False, device = None):
        if device is None:
            device = self.device
        _, x = self.model(xs, lengths)
        
        
        
        x = self.classification(x)
                    
        loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(device))
        
        return loss

    def forward(self, xs, lenghts, threshold = 0.4, device = None):
        if device is None:
            device = self.device
        # Get the emission scores from the BiLSTM
        _, x = self.model(xs, lenghts)
        
        scores = self.classification(x)
        
        if self.th is not None:
            threshold = self.th
        tag_seq = self.softmax(scores)[:,:,1]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]

class TransformerCRF(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=6, 
                 nheads = 8, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, positional_encoding = True, restricted = False, window_size = None):
        super(TransformerCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = Transformer(in_dim = embedding_dim, h_dim = hidden_dim, n_heads = nheads, n_layers = num_layers, dropout=dropout_in, drop_out = dropout_out, batch_first = batch_first, device = self.device, positional_encoding = positional_encoding, restricted = restricted, window_size = window_size)
        
        self.crf = CRF(embedding_dim, self.tagset_size)

    def loss(self, xs, lengths, tags, device = None):
        if device is None:
            device = self.device
        masks = create_mask(xs, lengths).to(device)
        out, features = self.model(xs, masks)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs, lenghts, device = None):
        if device is None:
            device = self.device
        # Get the emission scores from the Transformer
        masks = create_mask(xs, lenghts).to(device)
        out, features = self.model(xs, masks)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq

class Transformer_segmenter(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=6, 
                 nheads = 8, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, loss_fn = 'CrossEntropy', positional_encoding = True, threshold = None,
                 restricted = False, window_size = None, add_embedding = False):
                     
        """
        Transformer text segmenter that should replicate the sentence-level transformer of the transformer over transformer
        paper. There are also a series of options, such as final embeddings addition, that are not present in the original paper.
        TODO: add auxiliary coherence score
        """
        super(Transformer_segmenter, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size if loss_fn == "CrossEntropy" else 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        pyramidal = False
        self.no_mask = True # TODO: if everything works instantiating everything from huggingface, then remove the no_mask parameter and the condition of it being false in the forward method
        if restricted:
            pyramidal = True # TODO: disentangle pyramidal transformer from simple restricted one (give option to choose one or the other)
            self.no_mask = True
        
        if restricted:
            if pyramidal:
                window_size = [win*window_size for win in range(num_layers, 0, -1)]
            
            self.model = Longformer_Local_Attention(embedding_dim, nheads, num_layers, hidden_dim, window_size = window_size, dropout = dropout_in,
                 # activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps= 1e-5,
                 tagset_size = tagset_size,
                 device=None, max_position_embedding = 4096)
        
        else:
            self.model = Classic_Transformer(embedding_dim, nheads, num_layers, hidden_dim, dropout = dropout_in,
                 # activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps= 1e-5,
                 tagset_size = tagset_size,
                 device=None, max_position_embedding = 4096)
            #self.model = Transformer(in_dim = embedding_dim, h_dim = hidden_dim, n_heads = nheads, n_layers = num_layers, dropout=dropout_in, drop_out = dropout_out, batch_first = batch_first, positional_encoding = positional_encoding, device = self.device, restricted = restricted, window_size=window_size, pyramidal=pyramidal)
        
        self.add_emb = add_embedding
        final_dim_multiplier = 2 if self.add_emb else 1
        self.classification = nn.Linear(embedding_dim*final_dim_multiplier, self.tagset_size)
        
        if loss_fn == 'CrossEntropy':
            self.bce = False
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
        elif loss_fn == 'BinaryCrossEntropy':
            self.bce = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')
        
        self.softmax = nn.Softmax(dim=2)
        
        self.th = threshold
        
    def loss(self, xs, lengths, tags, device = None, aggregate_sentences = True):
        
        if device is None:
            device = self.device
        
        if self.no_mask:
            x = self.model(xs, lengths)
        else:
            masks = create_mask(xs, lengths).to(device)
            _, x = self.model(xs, masks)
        
        if aggregate_sentences:
            # Just an option to ignore padding completely with a brute force approach: in case of BCE loss this option might be prefer as there is no option to ignore the padded indeces in the loss (contrary to CE)
            outputs = []
            targets = []
            if self.add_emb:
                embs = []
            
            for index, length in enumerate(lengths):
                outputs.append(x[index][:length])
                targets.append(tags[index][:length])
                if self.add_emb:
                    embs.append(xs[index][:length])
            
            x = torch.cat(outputs, axis = 0).to(device)
            tags = torch.cat(targets, axis = 0)
            if self.add_emb:
                xs = torch.cat(embs, axis = 0).to(device)
        
        if self.add_emb:
            x = self.classification(torch.cat((x, xs), axis = 1)) if aggregate_sentences else self.classification(torch.cat((x, xs), axis = 2))
        else:
            x = self.classification(x)
        
                    
        if self.bce:
            x = self.sigmoid(x)
            loss = self.loss_fn(x.reshape(-1), tags.reshape(-1).to(device))
        else:
            loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(device))
        
        return loss

    def forward(self, xs, lenghts, threshold = 0.4, device = None):
        if device is None:
            device = self.device
        
        if self.no_mask:
            x = self.model(xs, lenghts)
        else:
            masks = create_mask(xs, lenghts).to(device)
            _, x = self.model(xs, masks)
        
        if self.add_emb:
            scores = self.classification(torch.cat((x, xs), axis = 1)) if aggregate_sentences else self.classification(torch.cat((x, xs), axis = 2))
        else:
            scores = self.classification(x)
        
        if self.th is not None:
            threshold = self.th
        
        if self.bce:
            scores = self.sigmoid(scores)
            tag_seq = scores[:,:,0]>threshold
        else:
            scores = self.softmax(scores)
            tag_seq = scores[:,:,1]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]
        
def double_masking(src, sentence_lengths, doc_lengths):
    batch_sentence, word_numbers, embed_size = src.shape
    batch = len(doc_lengths)
    
    mask = create_mask(src, sentence_lengths).to(src.device)
    src[mask] = torch.tensor(float('-inf')).to(src.device)
    
    src = src.contiguous().view(batch, -1, word_numbers, embed_size)
    mask = create_mask(src, doc_lengths).to(src.device)
    
    src[mask] = src.new_zeros(1)
    
    return src
    
def attention_pooling(src, context, sentence_lengths, doc_lengths = None):
    batch_sentence, word_numbers, embed_size = src.shape
    batch = len(doc_lengths)
    
    softmax = nn.Softmax(dim=1)
    
    """
    Below, we obtain the attention weights and then mask the padded elements with -inf so that the relative softmax score turns to 0 and we can multiply with the original input to obtained the contextualised sentence representation. 
    """
    attn_weights = src.matmul(context)
    mask = create_mask(src, sentence_lengths).to(src.device)
    
    attn_weights[mask] = torch.tensor(float('-inf')).to(src.device)
    attn_weights = softmax(attn_weights)
    
    src = attn_weights.unsqueeze(1).matmul(src).squeeze()
    
    if doc_lengths is not None:
        src = src.contiguous().view(batch, -1, word_numbers, embed_size)
        mask = create_mask(src, doc_lengths).to(src.device)
        
        src[mask] = src.new_zeros(1)
    
    return src
    

class TextSeg(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, Embedding_layer = None, num_layers=1, 
                 bidirectional = True, dropout_in=0.0, 
                 dropout_out = 0.0, batch_first = True, LSTM = True,
                 loss_fn = 'CrossEntropy', threshold = None, pooling = 'max', 
                 auxiliary_coherence_original = False,
                 restricted_self_attention = False, 
                 additional_lstm = True):
        
        """
        The original textseg algorithm from Kosherek et al. 2017. A number of improvements are also included as optional, such as the restricted self attention mechanism and the auxiliary coherence tasks, all of which are admejorments proposed by xing et al. 2019.
        """
        
        # TODO: Add the option of conatenating BERT embeddings at the sentence level
        super(TextSeg, self).__init__()
        
        if Embedding_layer is not None:
            self.embedding = Embedding_layer
        else:
            self.embedding = None

        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim

        self.word_level = RNN(embedding_dim, hidden_dim, num_layers, tagset_size, 
                         bidirectional, dropout_in, dropout_out, batch_first = batch_first,
                         LSTM = LSTM)
        
        self.sentence_level = RNN(hidden_dim*2, hidden_dim, num_layers, tagset_size, 
                         bidirectional, dropout_in, dropout_out, batch_first = batch_first,
                         LSTM = LSTM)

        self.classification = nn.Linear(hidden_dim*2, self.tagset_size)
        
        if loss_fn == 'CrossEntropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
        elif loss_fn == 'BinaryCrossEntropy':
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')
        
        self.softmax = nn.Softmax(dim=2)
        self.th = threshold
        
        self.aux = False
        if auxiliary_coherence_original:
            self.aux = True
            self.aux_projection = nn.Linear(hidden_dim, hidden_dim)
            self.coherence_loss = nn.BCELoss()
        
        self.rsa = False
        if restricted_self_attention:
            self.rsa = True
            self.self_attn = nn.Linear(hidden_dim*4+1, 1)
            self.win_size = 3 # TODO: add the option of changing the window size
            self.model2 = None
            if additional_lstm:
                self.model2 = RNN(hidden_dim*4, hidden_dim, num_layers, tagset_size, 
                                 bidirectional, dropout_in, dropout_out, batch_first = batch_first,
                                 LSTM = LSTM)
            
        
        if pooling=='max':
            self.pooling = torch.nn.AdaptiveMaxPool3d((None, 1, hidden_dim*2))
            self.attn = False
        elif pooling=='attn':
            self.feedforward = nn.Sequential(nn.Linear(2*hidden, 2*hidden), nn.Tanh())
            self.context_vector = nn.Parameter(torch.Tensor(2*hidden))
            self.context_vector.data.normal_(0, 0.1)
            self.attn = True
        else:
            raise NotImplementedError
        
    def loss(self, x, sentence_lengths, doc_lengths, tags):
        """
        x.shape = (batch_size, max_sentence, max_words) if using embedding layer
        x.shape = (batch_size, max_sentence, max_words, embedding_dim) otherwise
        """
        
        
        if self.embedding is not None:
            x = self.embedding(x)
            max_words = x.shape[-1]
        else:
            max_words = x.shape[-2]

        x = x.view(-1, max_words, self.embedding_dim)

        _, x = self.word_level(x, sentence_lengths)
        
        if self.attn:
            x = self.feedforward(x)
            x = attention_pooling(x, self.context_vector, sentence_lengths, doc_lengths)
        else:
            x = double_masking(x, sentence_lengths, doc_lengths)
            x = self.pooling(x).squeeze(-2)

        _, x = self.sentence_level(x, doc_lengths)
        
        if self.aux:
            
            sim, sim_tags = auxiliary_coherence_function(x, tags, doc_lengths, self.aux_projection)
            
            coh_loss = self.coherence_loss(sim, sim_tags.to(sim.device))
            
        if self.rsa:
            x = xing_restricted_attention(x, doc_lengths, self.win_size, self.self_attn)
            if self.model2 is not None:
                _, x = self.model2(x, doc_lengths)

        y = self.classification(x)
        
        tags = tags.reshape(-1).type(torch.LongTensor).to(x.device)
        
        loss =  self.loss_fn(y.reshape(-1, self.tagset_size), tags)
        
        if self.aux:
            loss = 0.2*coh_loss + 0.8*loss
        
        return loss

    def forward(self, x, sentence_lengths, doc_lengths, threshold = 0.4):
        # Get the emission scores from the BiLSTM
        """
        x.shape = (batch_size, max_sentence, max_words) if using embedding layer
        x.shape = (batch_size, max_sentence, max_words, embedding_dim) otherwise
        """
        
        
        if self.embedding is not None:
            x = self.embedding(x)
            max_words = x.shape[-1]
        else:
            max_words = x.shape[-2]

        x = x.view(-1, max_words, self.embedding_dim)

        _, x = self.word_level(x, sentence_lengths)
        
        x = double_masking(x, sentence_lengths, doc_lengths)
        
        x = self.pooling(x).squeeze(-2)

        _, x = self.sentence_level(x, doc_lengths)
        
        if self.rsa:
            x = xing_restricted_attention(x, doc_lengths, self.win_size, self.self_attn)
            if self.model2 is not None:
                _, x = self.model2(x, doc_lengths)

        scores = self.softmax(self.classification(x).detach())
        if self.th is not None:
            threshold = self.th
        
        tag_seq = scores[:,:,1]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(doc_lengths)]
        
class MLP(nn.Module):
    """
    A simple linear projection/MLP to be used on top of cross-encoder to emulate the setting of Lukashenk 2020
    """
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=1, 
                 dropout_in=0.0, dropout_out = 0.0, batch_first = True, loss_fn = 'CrossEntropy', 
                 threshold = None):
        
        super(MLP, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size if loss_fn == "CrossEntropy" else 1
        # TODO: ADD DEVICE PARAMETER PASSED BY PYTORCH LIGHTNING
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if num_layers == 0:
            self.proj = nn.Linear(embedding_dim, self.tagset_size)
        else:
            od = OrderedDict([("proj_0", nn.Linear(embedding_dim, hidden_dim)), ("relu_0", nn.ReLU())])
            for i in range(num_layers-1):
                od["proj_"+str(i+1)] = nn.Linear(hidden_dim, hidden_dim)
                od["relu_"+str(i+1)] = nn.ReLU()
                
            od["final_proj"] = nn.Linear(hidden_dim, self.tagset_size)

        self.model = nn.Sequential(od)
        
        if loss_fn == 'CrossEntropy':
            self.bce = False
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
        elif loss_fn == 'BinaryCrossEntropy':
            self.bce = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')
        
        self.th = threshold
        
    def loss(self, xs, lengths, tags, segments = None, auxiliary_task = False, use_cosine = True, aggregate_sentences = False, device = None):
        if device is None:
            device = self.device
        
        x = self.model(xs)
        
        if self.bce:
            x_unpad = []
            for i, x_i in enumerate(x):
                x_unpad.append(x_i[:lengths[i]])
            
            x = torch.concatenate(x_unpad, axis = 0)
            
            x = self.sigmoid(x)
            loss = self.loss_fn(x.reshape(-1), tags.reshape(-1).to(device))
        else:
            loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(device))
        
        return loss

    def forward(self, xs, lenghts, threshold = 0.4, device = None):
        if device is None:
            device = self.device
        # Get the emission scores from the BiLSTM
        x = self.model(xs)
        
        if self.th is not None:
            threshold = self.th
        
        if self.bce:
            tag_seq = self.sigmoid(scores)[:,:,0]>threshold
        else:
            tag_seq = self.softmax(scores)[:,:,1]>threshold
        
        return x, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]
        
class Pair_MLP(nn.Module):
    """
    A simple linear projection/MLP to be used on top of cross-encoder to emulate the setting of Lukashenk 2020
    """
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=1, 
                 dropout_in=0.0, dropout_out = 0.0, batch_first = True, loss_fn = 'CrossEntropy', 
                 threshold = None):
        
        super(Pair_MLP, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size if loss_fn == "CrossEntropy" else 1
        # TODO: ADD DEVICE PARAMETER PASSED BY PYTORCH LIGHTNING
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if num_layers == 0:
            od = OrderedDict([("dropout_in", nn.Dropout(p=dropout_in)), ("final_proj", nn.Linear(embedding_dim*3, self.tagset_size))])
        else:
            od = OrderedDict([("dropout_in_0", nn.Dropout(p=dropout_in)), ("proj_0", nn.Linear(embedding_dim, hidden_dim)), ("relu_0", nn.ReLU()), ("dropout_out_0", nn.Dropout(p=dropout_out))])
            for i in range(num_layers-1):
                od["dropout_in"+str(i+1)] = nn.Dropout(p=dropout_in)
                od["proj_"+str(i+1)] = nn.Linear(hidden_dim, hidden_dim)
                od["relu_"+str(i+1)] = nn.ReLU()
                od["dropout_out"+str(i+1)] = nn.Dropout(p=dropout_out)
            od["final_proj"] = nn.Linear(hidden_dim, self.tagset_size)

        self.model = nn.Sequential(od)
        
        self.fl = False
        if loss_fn == 'CrossEntropy':
            self.bce = False
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
        elif loss_fn == 'BinaryCrossEntropy':
            self.bce = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
        elif loss_fn == "FocalLoss":
            self.bce = True
            self.fl = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = sigmoid_focal_loss
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')
        
        self.th = threshold
        
    def loss(self, xs, lengths, tags, segments = None, auxiliary_task = False, use_cosine = True, aggregate_sentences = False, device = None):
        if device is None:
            device = self.device
        
        tags = 1-tags
        
        input_pairs = torch.cat([xs[:,:-1,:], xs[:,1:,:], torch.abs(xs[:,:-1,:]-xs[:,1:,:])], axis = 2)
        
        x = self.model(input_pairs)
        
        if self.bce:
            x_unpad, y_unpad = [], []
            for i, x_i in enumerate(x):
                x_unpad.append(x_i[:lengths[i]-1])
                y_unpad.append(tags[i][:lengths[i]-1])
            
            x = torch.cat(x_unpad, axis = 0)
            tags = torch.cat(y_unpad, axis = 0)
            
            if not self.fl:
                x = self.sigmoid(x)
            loss = self.loss_fn(x.reshape(-1), tags.reshape(-1).to(device))
        else:
            loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(device))
        
        return loss

    def forward(self, xs, lenghts, threshold = 0.4, device = None):
        if device is None:
            device = self.device
        # Get the emission scores from the BiLSTM
        bs, sl, hs = xs.shape
        input_pairs = torch.cat([xs[:,:-1,:], xs[:,1:,:], torch.abs(xs[:,:-1,:]-xs[:,1:,:])], axis = 2)
        
        scores = self.model(input_pairs)
        if self.bce:
            scores = torch.cat((scores, torch.ones(bs,1, 1).to(xs.device)), axis = 1)
        else:
            scores = torch.cat((scores, torch.ones(bs,1, 2).to(xs.device)), axis = 1)
        
        if self.th is not None:
            threshold = self.th
        
        if self.bce:
            scores = 1-self.sigmoid(scores)
            tag_seq = scores[:,:,0]>threshold
        else:
            scores = 1-self.softmax(scores)
            tag_seq = scores[:,:,1]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]


class BiLSTM_Transformer(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=1,
                 bidirectional=True, dropout_in=0.0,
                 dropout_out=0.0, batch_first=True, LSTM=True,
                 loss_fn='CrossEntropy', threshold=None, add_embedding=False,
                 auxiliary_coherence_original=False, restricted = True, window_size = 3,
                 parallel = True, nheads = 4, positional_encoding = True):

        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size if loss_fn == "CrossEntropy" else 1
        self.add_emb = add_embedding
        # TODO: ADD DEVICE PARAMETER PASSED BY PYTORCH LIGHTNING
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lstm = RNN(embedding_dim, hidden_dim, num_layers, tagset_size,
                         bidirectional, dropout_in, dropout_out, batch_first=batch_first,
                         LSTM=LSTM)

        self.parallel = False
        if parallel:
            self.parallel = True
            transformer_in = embedding_dim
            classifier_in = hidden_dim*2 + embedding_dim
        else:
            transformer_in = hidden_dim*2
            classifier_in = embedding_dim

        if add_embedding:
            classifier_in+=embedding_dim


        pyramidal = False
        if restricted:
            pyramidal = False  # TODO: try out pyramidal option instead

        self.transformer = Transformer(in_dim=transformer_in, h_dim=hidden_dim, n_heads=nheads, n_layers=num_layers,
                                 dropout=dropout_in, drop_out=dropout_out, batch_first=batch_first,
                                 positional_encoding=positional_encoding, device=self.device, restricted=restricted,
                                 window_size=window_size, pyramidal=pyramidal)

        self.classification = nn.Linear(classifier_in, self.tagset_size)

        if loss_fn == 'CrossEntropy':
            self.bce = False
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        elif loss_fn == 'BinaryCrossEntropy':
            self.bce = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')

        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.coherence_loss = nn.BCELoss()
        self.aux = False
        if auxiliary_coherence_original:
            self.aux = True
            self.aux_projection = nn.Linear(hidden_dim, hidden_dim)

        self.coherence_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid())

        self.softmax = nn.Softmax(dim=2)
        self.th = threshold

    def loss(self, xs, lengths, tags, segments=None, auxiliary_task=False, use_cosine=True, aggregate_sentences=True,
             device=None):
        if device is None:
            device = self.device
        _, x = self.lstm(xs, lengths)

        if self.aux:

            sim, sim_tags = auxiliary_coherence_function(x, tags, lengths, self.aux_projection)

            coh_loss = self.coherence_loss(sim, sim_tags.to(sim.device))


        elif segments is not None:
            # This option represent a series of experiments to substitute the auxiliary task by Carenini et al. No success so far.

            if use_cosine:
                coh_loss = cosine_loss(x, lengths, segments, self.cosine_loss, device)
            else:
                coh_loss = coherence_classification(x, lengths, segments, self.coherence_loss,
                                                    self.coherence_projection, device)


        if self.parallel:
            masks = create_mask(xs, lengths).to(device)
            _, x2 = self.transformer(xs, masks)

            x = torch.cat((x, x2), axis=2)
        else:
            masks = create_mask(x, lengths).to(device)
            _, x = self.transformer(x, masks)

        if aggregate_sentences:
            outputs = []
            targets = []
            if self.add_emb:
                embs = []

            for index, length in enumerate(lengths):
                outputs.append(x[index][:length])
                targets.append(tags[index][:length])
                if self.add_emb:
                    embs.append(xs[index][:length])

            x = torch.cat(outputs, axis=0).to(device)
            tags = torch.cat(targets, axis=0)
            if self.add_emb:
                xs = torch.cat(embs, axis=0).to(device)

        if self.add_emb:
            x = self.classification(torch.cat((x, xs), axis=1)) if aggregate_sentences else self.classification(
                torch.cat((x, xs), axis=2))
        else:
            x = self.classification(x)

        if self.bce:
            x = self.sigmoid(x)
            loss = self.loss_fn(x.reshape(-1), tags.reshape(-1).to(device))
        else:
            loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(device))

        if self.aux or segments is not None:
            loss = coh_loss * 0.2 + 0.8 * loss

        return loss

    def forward(self, xs, lenghts, threshold=0.4, device=None):
        if device is None:
            device = self.device
        # Get the emission scores from the BiLSTM
        _, x = self.model(xs, lenghts)

        if self.parallel:
            masks = create_mask(xs, lengths).to(device)
            _, x2 = self.transformer(xs, masks)

            x = torch.cat((x, x2), axis=2)
        else:
            masks = create_mask(x, lengths).to(device)
            _, x = self.transformer(x, masks)

        if self.add_emb:
            scores = self.classification(torch.cat((x, xs), axis=1)) if aggregate_sentences else self.classification(
                torch.cat((x, xs), axis=2))
        else:
            scores = self.classification(x)

        if self.th is not None:
            threshold = self.th

        if self.bce:
            scores = self.sigmoid(scores)
            tag_seq = scores[:, :, 0] > threshold
        else:
            scores = self.softmax(scores)
            tag_seq = scores[:, :, 1] > threshold

        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]

class SheikhBiLSTM(nn.Module):
    def __init__(self,
    tagset_size, embedding_dim, hidden_dim, num_layers, dropout_in=0.5, 
    dropout_attention = 0.0, batch_first = True, loss_fn = 'BinaryCrossEntropy', threshold = None):
        super(SheikhBiLSTM, self).__init__()
        self.lstm = RNN(embedding_dim, hidden_dim, num_layers, tagset_size, 
                         True, dropout_in, dropout_attention, batch_first = batch_first,
                         LSTM = True)

        self.forward_dense = nn.Linear(hidden_dim, hidden_dim)
        self.backward_dense = nn.Linear(hidden_dim, hidden_dim)

        if loss_fn == 'BinaryCrossEntropy':
            self.fl = False
            self.loss_fn = nn.BCELoss()
            self.classification = nn.Linear(hidden_dim*2, 1)
        elif loss_fn == 'FocalLoss':
            self.fl = True
            self.loss_fn = sigmoid_focal_loss
            self.classification = nn.Linear(hidden_dim*2, 1)
        else:
            raise ValueError('Choose one of FocalLoss or BinaryCrossEntropy as loss function')

        self.sigmoid = nn.Sigmoid()
        self.th = threshold

    
    def loss(self, inputs, lengths, tags, device = None, segments = None):
        _, x = self.lstm(inputs, lengths)
        tags = 1 - tags
        bs, sl, hs = x.shape
        x = x.view(bs, sl, 2, -1)
        x_for = x[:,:-1,0,:]
        x_bac = x[:,1:,1,:]
        x_for = self.forward_dense(x_for)
        x_bac = self.backward_dense(x_bac)
        x = (x_for*x_bac).sum(axis=2, keepdims = True)
        x = self.sigmoid(x)
        x_unpad, y_unpad = [], []
        for i, x_i in enumerate(x):
            x_unpad.append(x_i[:lengths[i]-1])
            y_unpad.append(tags[i][:lengths[i]-1])
        
        if self.fl:
            loss = self.loss_fn(torch.cat(x_unpad).reshape(-1), torch.cat(y_unpad).to(inputs.device), sigmoid = False)
        else:
            loss = self.loss_fn(torch.cat(x_unpad).reshape(-1), torch.cat(y_unpad).to(inputs.device))
        return loss
        
    def forward(self, inputs, lenghts, threshold = 0.4, device = None):
        # Get the emission scores from the BiLSTM
        _, x = self.lstm(inputs, lenghts)
        bs, sl, hs = x.shape
        x = x.view(bs, sl, 2, -1)
        x_for = x[:,:-1,0,:]
        x_bac = x[:,1:,1,:]
        x_for = self.forward_dense(x_for)
        x_bac = self.backward_dense(x_bac)
        scores = (x_for*x_bac).sum(axis=2, keepdims = True)
        scores = torch.cat((scores, torch.ones(bs,1, 1).to(inputs.device)), axis = 1) # Add the extra time step corresponding to the last output (will affect just the longest sequence)
        if self.th is not None:
            threshold = self.th
        scores = 1-self.sigmoid(scores)
        tag_seq = scores[:,:,0]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lenghts)]
        
class SheikhTransformer(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, num_layers=6, 
                 nheads = 8, dropout_in=0.0, loss_fn = 'CrossEntropy', threshold = None,
                 window_size = None):
        super(SheikhTransformer, self).__init__()
        self.transformer_for = Causal_Transformer(embedding_dim, nheads, num_layers, hidden_dim,  
                         dropout_in, layer_norm_eps = 1e-12,
                         tagset_size = 2,
                         device=None, max_position_embedding = 1024, backward = False, window = window_size)
        
        self.transformer_bac = Causal_Transformer(embedding_dim, nheads, num_layers, hidden_dim,  
                         dropout_in, layer_norm_eps = 1e-12,
                         tagset_size = 2,
                         device=None, max_position_embedding = 1024, backward = True, window = window_size)

        self.forward_dense = nn.Linear(embedding_dim, hidden_dim)
        self.backward_dense = nn.Linear(embedding_dim, hidden_dim)

        if loss_fn == 'BinaryCrossEntropy':
            self.fl = False
            self.loss_fn = nn.BCELoss()
            self.classification = nn.Linear(hidden_dim*2, 1)
        elif loss_fn == 'FocalLoss':
            self.fl = True
            self.loss_fn = sigmoid_focal_loss
            self.classification = nn.Linear(hidden_dim*2, 1)
        else:
            raise ValueError('Choose one of FocalLoss or BinaryCrossEntropy as loss function')

        self.sigmoid = nn.Sigmoid()
        self.th = threshold

    
    def loss(self, inputs, lengths, tags, device = None, segments = None):
        tags = 1 - tags
        x_for = self.transformer_for(inputs, lengths)
        x_bac = self.transformer_bac(inputs, lengths)
        bs, sl, hs = x_for.shape
        x_for = self.forward_dense(x_for)
        x_bac = self.backward_dense(x_bac)
        x = (x_for*x_bac).sum(axis=2, keepdims = True)
        x = self.sigmoid(x)
        x_unpad, y_unpad = [], []
        for i, x_i in enumerate(x):
            x_unpad.append(x_i[:lengths[i]-1])
            y_unpad.append(tags[i][:lengths[i]-1])
        
        if self.fl:
            loss = self.loss_fn(torch.cat(x_unpad).reshape(-1), torch.cat(y_unpad).to(inputs.device), sigmoid = False)
        else:
            loss = self.loss_fn(torch.cat(x_unpad).reshape(-1), torch.cat(y_unpad).to(inputs.device))
        return loss
        
    def forward(self, inputs, lengths, threshold = 0.4, device = None):
        # Get the emission scores from the BiLSTM
        x_for = self.transformer_for(inputs, lengths)
        x_bac = self.transformer_bac(inputs, lengths)
        bs, sl, hs = x_for.shape
        x_for = self.forward_dense(x_for)
        x_bac = self.backward_dense(x_bac)
        scores = (x_for*x_bac).sum(axis=2, keepdims = True)
        scores = torch.cat((scores, torch.ones(bs,1, 1).to(inputs.device)), axis = 1) # Add the extra time step corresponding to the last output (will affect just the longest sequence)
        if self.th is not None:
            threshold = self.th
        scores = 1-self.sigmoid(scores)
        tag_seq = scores[:,:,0]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length.data] for index, length in enumerate(lengths)]


class CrossEncoderModel(nn.Module):
    def __init__(self, tagset_size, encoder, pool = "cls", hidden_dim='auto', loss_fn = 'CrossEntropy', threshold = 0.5):
        
        super(CrossEncoderModel, self).__init__()
        self.bert = AutoModel.from_pretrained(encoder)
        self.tagset_size = tagset_size
        if hidden_dim=='auto':
            hidden_dim = self.bert.config.hidden_size
        self.classification = nn.Linear(hidden_dim, self.tagset_size)
        
        self.fl = False
        if loss_fn == 'CrossEntropy':
            self.bce = False
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
        elif loss_fn == 'BinaryCrossEntropy':
            0/0
            self.bce = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = nn.BCELoss()
        elif loss_fn == "FocalLoss":
            self.bce = True
            self.fl = True
            self.sigmoid = nn.Sigmoid()
            self.loss_fn = sigmoid_focal_loss
        else:
            raise ValueError('Choose one of CrossEntropy or BinaryCrossEntropy as loss function')
        
        self.max_seq_len = 150
        self.softmax = nn.Softmax(dim=1)
        self.th = threshold
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

    def loss(self, xs, attention_mask, tags, device = None):
        if device is None:
            device = self.device
        lengths, attention_mask = attention_mask
        x = []
        targets = []
        for i, x_ in enumerate(xs):
            x_ = xs[i, :lengths[i]]
            seq_len = x_.shape[0]
            print(x_.shape)
            if seq_len>self.max_seq_len:
                x_tmp = []
                
                for idx in range(seq_len//self.max_seq_len):
                    x_tmp.append(self.bert(input_ids=x_[self.max_seq_len*idx:self.max_seq_len*(idx+1)], attention_mask=attention_mask[i, self.max_seq_len*idx:self.max_seq_len*(idx+1)], output_hidden_states=True).hidden_states)
                if seq_len%self.max_seq_len:
                    x_tmp.append(self.bert(input_ids=x_[self.max_seq_len*(idx+1):seq_len], attention_mask=attention_mask[i, self.max_seq_len*(idx+1):seq_len], output_hidden_states=True).hidden_states)
                x_ = [torch.cat([e[i] for e in x_tmp], axis = 0) for i in range(len(x_tmp[0]))]
            else:
                x_ = self.bert(input_ids=x_, attention_mask=attention_mask[i, :lengths[i]], output_hidden_states=True).hidden_states
            #print(x_[-1].shape)
            
            x.append(self.pool(x_, attention_mask[i, :lengths[i]]))
            targets.append(tags[i][:lengths[i]])
            
        x = torch.cat(x, axis = 0)
        tags = torch.cat(targets, axis = 0)
        
        x = self.classification(x)
        
                    
        if self.bce:
            if not self.fl:
                x = self.sigmoid(x)
            loss = self.loss_fn(x.reshape(-1), tags.reshape(-1).to(device))
        else:
            loss = self.loss_fn(x.reshape(-1, self.tagset_size), tags.reshape(-1).type(torch.LongTensor).to(device))
        
        return loss

    def forward(self, xs, attention_mask, threshold = 0.4, device = None):
        if device is None:
            device = self.device
        lengths, attention_mask = attention_mask
        bs, max_seq_len, ws = xs.shape
        x = []
        for i, x_ in enumerate(xs):
            x_ = xs[i, :lengths[i]]
            seq_len = x_.shape[0]
            if seq_len>self.max_seq_len:
                x_tmp = []
                
                for idx in range(seq_len//self.max_seq_len):
                    x_tmp.append(self.bert(input_ids=x_[self.max_seq_len*idx:self.max_seq_len*(idx+1)], attention_mask=attention_mask[i, self.max_seq_len*idx:self.max_seq_len*(idx+1)], output_hidden_states=True).hidden_states)
                if seq_len%self.max_seq_len:
                    x_tmp.append(self.bert(input_ids=x_[self.max_seq_len*(idx+1):], attention_mask=attention_mask[i, self.max_seq_len*(idx+1):], output_hidden_states=True).hidden_states)
                x_ = torch.cat(x_tmp, axis = 0)
            else:
                x_ = self.bert(input_ids=x_, attention_mask=attention_mask[i, :lengths[i]], output_hidden_states=True).hidden_states
            print(x_[-1].shape)
            
            x.append(self.pool(x_, attention_mask[i, :lengths[i]]))
        x = torch.cat(x)
        
        scores = self.classification(x)
        print(scores.shape)
        
        if self.th is not None:
            threshold = self.th
        
        if self.bce:
            new_scores = torch.zeros((bs, max_seq_len, 1))
            idx=0
            for i, length in enumerate(lengths):
                new_scores[i, :length] = scores[idx:idx+length]
                idx = length
            scores = self.sigmoid(new_scores)
            tag_seq = scores[:,:,0]>threshold
        else:
            new_scores = torch.zeros((bs, max_seq_len, 2))
            idx=0
            for i, length in enumerate(lengths):
                new_scores[i, :length] = scores[idx:idx+length]
                idx = length
            scores = self.softmax(new_scores)
            tag_seq = scores[:,:,1]>threshold
        
        return scores, [tag_seq[index].detach().tolist()[:length] for index, length in enumerate(lengths)]