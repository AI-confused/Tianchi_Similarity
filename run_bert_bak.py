from __future__ import absolute_import
import argparse
import csv
import logging
import os
import random
import sys
from io import open
import pandas as pd
import numpy as np
import torch
import time
# from scipy.stats import spearmanr
import collections
import torch.nn as nn
from collections import defaultdict
import gc
import itertools
from multiprocessing import Pool
import functools
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from typing import Callable, Dict, List, Generator, Tuple
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
# from sklearn.metrics import f1_score
import json
import math
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, get_cosine_schedule_with_warmup
from model import BertForSimilary
# from clean import clean_data
from itertools import cycle

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = {
                'input_ids': choices_features[1],
                'input_mask': choices_features[2],
                'segment_ids': choices_features[3]
            }        
        self.label = label
        
def read_examples(input_file, is_training):
    df=pd.read_csv(input_file)
    if is_training==1 or is_training==2:
        examples=[]
        for val in df[['id','query1','query2','label']].values:
            examples.append(InputExample(guid=val[0],text_a=val[1],text_b=val[2],label=val[3]))
    else:
        examples=[]
        for val in df[['id','query1','query2']].values:
            examples.append(InputExample(guid=val[0],text_a=val[1],text_b=val[2]))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for example_index, example in enumerate(examples):

        query1_tokens=tokenizer.tokenize(example.text_a)
        query2_tokens=tokenizer.tokenize(example.text_b)

        _truncate_seq_pair(query1_tokens, query2_tokens, max_seq_length - 3)
        tokens = ["[CLS]"]+ query1_tokens + ["[SEP]"] + query2_tokens  + ["[SEP]"]
        segment_ids = [0] * (len(query1_tokens) + 2) + [1] * (len(query2_tokens) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)


        padding_length = max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        choices_features = (tokens, input_ids, input_mask, segment_ids)
        
        if is_training==1 or is_training==2:
            label = example.label
        else:
            label = 0
        if example_index < 1 and is_training==1:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581','_')))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
            logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            logger.info("label: {}".format(label))


        features.append(
            InputFeatures(
                example_id=example.guid,
                choices_features=choices_features,
                label=label
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def select_field(features, field):
    return [feature.choices_features[field] for feature in features]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def get_accuracy(predict, label):
    max_len = len(predict)
    acc = 0
    for i in range(max_len):
        if predict[i] == label[i]:
            acc += 1
    return round(acc/max_len, 4)
    
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
#             print(name, type(param),param)
            if param.requires_grad and emb_name in name:
#                 print(name)
                self.backup[name] = param.data.clone()
#                 print(param.grad)
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
# class PGD():
#     def __init__(self, model):
#         self.model = model
#         self.emb_backup = {}
#         self.grad_backup = {}

#     def attack(self, epsilon=1., alpha=0.3, emb_name='bert.embeddings.word_embeddings.weight', is_first_attack=False):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 if is_first_attack:
#                     self.emb_backup[name] = param.data.clone()
#                 norm = torch.norm(param.grad)
#                 if norm != 0 and not torch.isnan(norm):
#                     r_at = alpha * param.grad / norm
#                     param.data.add_(r_at)
#                     param.data = self.project(name, param.data, epsilon)

#     def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name: 
#                 assert name in self.emb_backup
#                 param.data = self.emb_backup[name]
#         self.emb_backup = {}

#     def project(self, param_name, param_data, epsilon):
#         r = param_data - self.emb_backup[param_name]
#         if torch.norm(r) > epsilon:
#             r = epsilon * r / torch.norm(r)
#         return self.emb_backup[param_name] + r

#     def backup_grad(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.grad_backup[name] = param.grad.clone()

#     def restore_grad(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 param.grad = self.grad_backup[name]
    
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--index", default=None, type=int, required=True,
                        help="")
    ## Other parameters
    parser.add_argument("--max_seq_length", default=32, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")  
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Set seed
    set_seed(args)


    try:
        os.makedirs(args.output_dir)
    except:
        pass
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    if args.do_train:
        # Prepare model
        model = BertForSimilary.from_pretrained(args.model_name_or_path, config=config)
#         fgm = FGM(model)
#         pgd = PGD(model)
#         K = 3
#         print(model)
        model.to(device)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
        train_examples = read_examples(os.path.join(args.data_dir, 'train.csv'), is_training = 1)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, 1)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer

        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        best_acc=0
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0        
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write('*'*80)
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
             # 正常训练
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            loss.backward()
            # 对抗训练
#             fgm.attack() # 在embedding上添加对抗扰动
#             loss_adv = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
#             if args.n_gpu > 1:
#                 loss_adv = loss_adv.mean() # mean() to average on multi-gpu.
#             if args.gradient_accumulation_steps > 1:
#                 loss_adv = loss_adv / args.gradient_accumulation_steps
#             loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#             fgm.restore() # 恢复embedding参数
            

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
#                 scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1


            if (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0 
                logger.info("***** Report result *****")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))


            if args.do_eval and (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                if args.do_eval_train:
                    file_list = ['train.csv','dev.csv']
                else:
                    file_list = ['dev.csv']
                for file in file_list:
                    inference_labels=[]
                    gold_labels=[]
                    inference_logits=[]
                    eval_examples = read_examples(os.path.join(args.data_dir, file), is_training = 2)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, 2)
                    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)                   


                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
                        
                    logger.info("***** Running evaluation *****")
                    logger.info("  Eval file = %s", file)
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)  
                        
                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)


                        with torch.no_grad():
                            tmp_eval_loss= model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        inference_labels.append(np.argmax(logits, axis=1))
                        gold_labels.append(label_ids)
#                         inference_logits.append(logits)
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1
                        
                    gold_labels=np.concatenate(gold_labels,0) 
                    inference_labels=np.concatenate(inference_labels,0)
                    model.train()
                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = get_accuracy(inference_labels, gold_labels)

                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'global_step': global_step,
                              'loss': train_loss}

                    if 'dev' in file:
                        with open(output_eval_file, "a") as writer:
                            writer.write(file+'\n')
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))
                            writer.write('*'*80)
                            writer.write('\n')
                    if eval_accuracy>best_acc and 'dev' in file:
                        print("="*80)
                        print("Best ACC",eval_accuracy)
                        print("Saving Model......")
                        best_acc=eval_accuracy
                        # Save a trained model
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("="*80)
                    else:
                        print("="*80)
        with open(output_eval_file, "a") as writer:
            writer.write('bert_acc: %f'%best_acc)
    
    if args.do_test:
        args.do_train=False
        model = BertForSimilary.from_pretrained(os.path.join(args.output_dir, "pytorch_model.bin"), config=config)

        model.to(device)

        test_file = '/tcdata/test.csv'
#         test_file = 'data/unknow_generate.csv'
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)        
        
        inference_labels=[]
        gold_labels=[]
        eval_examples = read_examples(test_file, is_training = 3)
        eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, 3)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
#         all_labels = torch.tensor([f.label for f in eval_features], dtype=torch.long)                   

        
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        for input_ids, input_mask, segment_ids in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask).detach().cpu().numpy()
                inference_labels.append(logits)
            
        logits=np.concatenate(inference_labels,0)
        df=pd.read_csv(test_file)
        df['logit_0']=logits[:,0]
        df['logit_1']=logits[:,1]

        work_dir = 'logit_result/result_%d.csv'%args.index
        df[['id', 'logit_0', 'logit_1']].to_csv(work_dir, index=False)
        logger.info('predict done')
            
if __name__ == "__main__":
    main()