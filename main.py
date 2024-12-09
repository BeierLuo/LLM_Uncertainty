import os
import torch
import torch.nn.functional as F
import evaluate
from datasets import load_metric
from tqdm import tqdm
import numpy as np
import pickle
from utils import get_llama_activations_bau, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q
import llama_iti
import pickle
import argparse
import matplotlib.pyplot as plt
from pprint import pprint
from baukit import Trace, TraceDict
from metric_utils import get_measures, print_measures
import re
from torch.autograd import Variable
from dataset import load_data
from utils import get_index, seed_everything
from algorithms import get_estimator
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa')
    parser.add_argument('--estimator_name', type=str, default='haloscope')
    parser.add_argument('--bleurt_model', type=str, default='lucadiliello/BLEURT-20-D12')
    parser.add_argument('--output_dir', type=str, default='/data/home/beier/output')
    parser.add_argument('--num_gene', type=int, default=1)
    parser.add_argument('--gene', type=int, default=0)
    parser.add_argument('--generate_gt', type=int, default=0)
    parser.add_argument('--use_rouge', type=int, default=0)
    parser.add_argument('--weighted_svd', type=int, default=0)
    parser.add_argument('--feat_loc_svd', type=int, default=0)
    parser.add_argument('--wild_ratio', type=float, default=0.75)
    parser.add_argument('--thres_gt', type=float, default=0.5, help='threshold of correct answer from unlabeled data')
    parser.add_argument('--most_likely', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument("--model_dir", type=str, default=None, help='local directory with model data')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    dataset = load_data(args)
    index_dict = get_index(args, dataset)

    estimator = get_estimator(args, dataset, index_dict)

    if args.gene:
        estimator.generate(args)  # generate answers for datasets
    elif args.generate_gt:
        estimator.evaluate_answers(args)
    else:
        estimator.estimate(args)
            

if __name__ == '__main__':
    seed_everything(42)
    main()