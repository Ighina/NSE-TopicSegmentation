import segeval as seg
import numpy as np


class Accuracy:
    def __init__(self, threshold=0.3):
        self.pk_to_weight = []
        self.windiff_to_weight = []
        self.threshold = threshold

    def update(self, h, gold, sentences_length = None):
        h_boundaries = self.get_seg_boundaries(h, sentences_length)
        gold_boundaries = self.get_seg_boundaries(gold, sentences_length)
        pk, count_pk = self.pk(h_boundaries, gold_boundaries, window_size = 10)
        windiff, count_wd = -1, 400;# self.win_diff(h_boundaries, gold_boundaries)

        if pk != -1:
            self.pk_to_weight.append((pk, count_pk))
        else:
            print ('pk error')

        if windiff != -1:
            self.windiff_to_weight.append((windiff, count_wd))

    def get_seg_boundaries(self, classifications, sentences_length = None):
        """
        :param list of tuples, each tuple is a sentence and its class (1 if it the sentence starts a segment, 0 otherwise).
        e.g: [(this is, 0), (a segment, 1) , (and another one, 1)
        :return: boundaries of segmentation to use for pk method. For given example the function will return (4, 3)
        """
        curr_seg_length = 0
        boundaries = []
        
        classifications[-1] = 1
        
        for i, classification in enumerate(classifications):
            is_split_point = bool(classifications[i])
            add_to_current_segment = 1 if sentences_length is None else sentences_length[i]
            curr_seg_length += add_to_current_segment
            if (is_split_point):
                boundaries.append(curr_seg_length)
                curr_seg_length = 0

        return boundaries

    def pk(self, h, gold, window_size=-1):
        """
        :param gold: gold segmentation (item in the list contains the number of words in segment) 
        :param h: hypothesis segmentation  (each item in the list contains the number of words in segment)
        :param window_size: optional 
        :return: accuracy
        """
        if window_size != -1:
            false_seg_count, total_count = seg.pk(h, gold, window_size=window_size, return_parts=True)
        else:
            false_seg_count, total_count = seg.pk(h, gold, return_parts=True)

        if total_count == 0:
            # TODO: Check when happens
            false_prob = -1
        else:
            false_prob = float(false_seg_count) / float(total_count)

        return false_prob, total_count

    def win_diff(self, h, gold, window_size=-1):
        """
        :param gold: gold segmentation (item in the list contains the number of words in segment) 
        :param h: hypothesis segmentation  (each item in the list contains the number of words in segment)
        :param window_size: optional 
        :return: accuracy
        """
        if window_size != -1:
            false_seg_count, total_count = seg.window_diff(h, gold, window_size=window_size, return_parts=True)
        else:
            false_seg_count, total_count = seg.window_diff(h, gold, return_parts=True)

        if total_count == 0:
            false_prob = -1
        else:
            false_prob = float(false_seg_count) / float(total_count)

        return false_prob, total_count

    def calc_accuracy(self):
        pk = sum([pw[0] * pw[1] for pw in self.pk_to_weight]) / sum([pw[1] for pw in self.pk_to_weight]) if len(
            self.pk_to_weight) > 0 else -1.0
        windiff = sum([pw[0] * pw[1] for pw in self.windiff_to_weight]) / sum(
            [pw[1] for pw in self.windiff_to_weight]) if len(self.windiff_to_weight) > 0 else -1.0

        return pk, windiff
        
        
def get_boundaries(boundaries):
    tot_sents = 0
    masses = []
    for boundary in boundaries:
        tot_sents += 1
        if boundary:
            masses.append(tot_sents)
            tot_sents = 0
    return masses

def get_k(refs):
    '''
    k : Average segment size in reference
    '''
    ref_count = 0
    total_count = 0
    for ref in refs:
        if ref == 1:
            ref_count += 1
        total_count += 1
    if ref_count == 0:
        k = 1
    else:
        k = int(round((total_count-ref_count)/(ref_count * 2.)))
    return k

def compute_Pk(boundaries, ground_truth, window_size = 'auto', boundary_symb = '1'):
    boundaries[-1] = 1
    ground_truth[-1] = 1
    h = get_boundaries(boundaries)
    t = get_boundaries(ground_truth)
    
    if window_size is None:
        result = seg.pk(h, t) # The paper from Kelvin et al. actually used window_size = 10! This definetely skew the whole results. Trying here by changing the window_size value manually
    else:
        if window_size=="auto":
            k = get_k(t)
        elif window_size>0:
            k = int(window_size)
        else:
            raise ValueError("Window size for Pk metric needs to be either integer, 'auto' or None")
        result = seg.pk(h,t, window_size = k) # The paper Topic Segmentation Model Focusing on Local Context use approximately (they subtract the final sentences of each segment) half the average segmnent length as k, which is 3 for wikicity. However, they concatenate all the references and predictions before computing pk, resulting in considering all terminal sentences as correct predictions
    
    boundaries[-1] = 0
    ground_truth[-1] = 0
    return result


def compute_window_diff(boundaries, ground_truth, window_size = 'auto', 
                            segval = True, boundary_symb = '1'):
    boundaries[-1] = 1
    ground_truth[-1] = 1
    h = get_boundaries(boundaries)
    t = get_boundaries(ground_truth)
    
    if window_size is None:
        result = seg.window_diff(h, t)
    else:
        if window_size=="auto":
            k = get_k(t)
        elif window_size>0:
            k = int(window_size)
        else:
            raise ValueError("Window size for Window Difference metric needs to be either a positive integer, 'auto' or None")
        result = seg.window_diff(h,t, window_size = k)
    
    boundaries[-1] = 0
    ground_truth[-1] = 0
    return result


def WinPR(reference, hypothesis, k = 10):
    """
    Implementation of the metric by scaiano et al. 2012 (https://aclanthology.org/N12-1038.pdf)
    
    Parameters
    ----------
    reference : list of int
        the reference segmentation (e.g. [0,0,0,1,0,0).
    hypothesis : list of int
        the hypothesised segmentation (e.g. [0,0,1,0,0,0]).
    k : int, optional
        The window value as defined in scaiano et al. 2012. The default is 10.

    Returns
    -------
    Precision, Recall and F1 measures (floats).

    """
    assert len(reference)==len(hypothesis), "Hypothesis and reference should be the same length!"
    
    N = len(reference)
    
    RC = []
    Spans_R = []
    Spans_C = []
    
    for i in range(1-k, N+1):
        prev_br = 0
        prev_bc = 0
        
        try:
            if Spans_R[-1][0] == 1:
                prev_br = 1
        except IndexError:
            pass
        try:
            if Spans_C[-1][0] == 1:
                prev_bc = 1
        except IndexError:
            pass
        
        Spans_R.append(reference[i:i+k])
        Spans_C.append(hypothesis[i:i+k])
        
        R = sum(reference[max(i,0):i+k])+prev_br
        C = sum(hypothesis[max(i,0):i+k]) + prev_bc
        
        RC.append((R,C))
    
    # RC = [(sum(reference[i:i+k]), sum(hypothesis[i:i+k])) for i in range(1-k, N+1)]
    
    TP =  sum([min(R, C) for R, C in RC])
    
    TN = -k*(k-1) + sum([k - max(R, C) for R, C in RC])
    
    FP = sum([max(0, C - R) for R, C in RC])
    
    FN = sum([max(0, R - C) for R, C in RC])
    try:
        precision = TP/(TP+FP)
    except ZeroDivisionError:
        return 0, 0, 0
        
    recall = TP/(TP+FN)
    
    f1 = 2*(precision*recall/(precision+recall))
    
    return precision, recall, f1


def B_measure(boundaries, ground_truth):
    """
    Boundary edit distance-based methods for text segmentation evaluation (Fournier2013)
    """
    boundaries[-1] = 1
    ground_truth[-1] = 1
    h = get_boundaries(boundaries)
    t = get_boundaries(ground_truth)
    
    try:     
        cm = seg.boundary_confusion_matrix(h, t)
    
        b_precision = seg.precision(cm)
        b_recall = seg.recall(cm)
        b_f1 = seg.fmeasure(cm)
    
    except ValueError:
        b_precision = 0
        b_recall = 0
        b_f1 = 0
    
    try:
        b = seg.boundary_similarity(h, t)
    except ValueError:
        b = 0
    
    return float(b_precision), float(b_recall), float(b_f1), float(b)