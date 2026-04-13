from sacrebleu.metrics import BLEU
from rouge import Rouge
import bert_score
import os
import pandas as pd
from tqdm import tqdm
import logging, json
import numpy as np

logging.basicConfig(level=logging.ERROR)

def get_bleu(hyp, ref):
    bleu = BLEU()
    return bleu.sentence_score(hyp, ref)._verbose

def get_rouge(hyp, ref, type='l'):
    rouge = Rouge()
    return rouge.get_scores(hyp, ref)[0][f'rouge-{type}']

def get_bertscore(hyp, ref):
    scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    
    p, r, f1 = scorer.score(hyp, ref, verbose=True)

    return f1.tolist()