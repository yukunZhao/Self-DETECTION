import torch
import numpy as np
import sys
import os
import requests
import json
import argparse
import random
from fastchat.model import load_model, get_conversation_template, add_model_args
from fastchat.model.model_adapter import get_conversation_template

from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('../all-MiniLM-L6-v2')
from numpy import dot
from numpy.linalg import norm

import math


from rouge import Rouge
# Utils functions
def read_json_sample_line(filename):
    data = []
    for line in open(filename):
        data.append(json.loads(line.strip()))
    return data

def read_sample_line(filename):
    data = []
    for line in open(filename):
        data.append(line.strip())
    return data

def extract_elements_from_json_data(data, key):
    elements = []
    for item in data:
        elements.append(item[key])
    return elements

# 计算 entropy, 输入是二维数组, self-consistency / consistency score
def cal_entropy(response_list):
    entropy_list = []
    question_cnt = 0
    for idx, answers_tmp in enumerate(response_list):
        question_cnt += len(answers_tmp)
        entropy_list.append(len(answers_tmp))
    entropy_list = [item * 1.0 / question_cnt for item in entropy_list]

    my_entropy = 0.0
    for p_tmp in entropy_list:
        my_entropy += -1.0 * p_tmp * math.log(p_tmp)
    return my_entropy, max(entropy_list)

# selfCheckGPT, part 3 of 3 , 计算原始的response和任意采样的response的contracted score
def cal_inter_contradict(response_list):
    orig_res = response_list[0][0]
    question_cnt = 0
    for idx, answers_tmp in enumerate(response_list):
        question_cnt += len(answers_tmp)
    contradicted_answers_cnt = 0
    if len(response_list) > 0:
        for answers in response_list[1:]:
            contradicted_answers_cnt += len(answers)
    return contradicted_answers_cnt * 1.0 / question_cnt

def cosine_simlarity(e1, e2):
    assert(len(e1) == len(e2))
    return dot(e1, e2)/ (norm(e1) * norm(e2))

# 计算原始的response和 一个answer的 bert sim score
def get_bert_sim_i(ori_answer, new_answer, ori_answer_embs=[]):
    if len(ori_answer_embs) == 0:
        ori_answer_embs = bert_model.encode(ori_answer)
    current_sim = 0.0
    for answer_ref in new_answer.split('\n'):
        answer_ref = answer_ref.strip()
        if len(answer_ref) > 0:
            sent_bert_embeddings = bert_model.encode(answer_ref)
            sim_tmp = cosine_simlarity(ori_answer_embs, sent_bert_embeddings)
            if sim_tmp > current_sim:
                current_sim = sim_tmp
    return current_sim

# selfCheckGPT, 1st of 3 parts, 通过bert_score 计算 任务两个response之间的相似度
def cal_bert_contradict_score(response_list):
    orig_res = response_list[0][0]
    bert_sim_score_list = []
    answers = []
    for idx, answers_tmp in enumerate(response_list):
        answers.extend(answers_tmp)
    ori_answer_embs = bert_model.encode(orig_res)
    bert_sim_scores = []
    for comp_ans in answers[1:]:
        bert_sim_score = get_bert_sim_i(orig_res, comp_ans, ori_answer_embs)
        bert_sim_scores.append(bert_sim_score)

    return 1.0 - sum(bert_sim_scores) / len(bert_sim_scores)

# selfCheckGPT, 2nd of 3 parts, n-gram
def cal_neglog_probs(logits):
    #logits_np = [math.exp(item) for item in logits]
    return -1.0 *sum(logits) / len(logits), -1.0 * max(logits)


# input: 经过norm后的logits
def cal_token_probs(logits):
    logits_np = [math.exp(item) for item in logits]
    return sum(logits_np) / len(logits_np), max(logits_np), min(logits_np)

def get_perplexity(logprobs):
    if len(logprobs) == 0:
        return -1
    s = sum(logprobs) / len(logprobs)
    return math.exp(-1*s) #math.pow(2, -1 * s), -1 *s, 

# 计算的特征包含：
# 1. atypicality,
# 2. consistency between phrases: entropy, max_ratio
# 3. token_probs
# 4. atypicality_avg
# 5. perplexity,
# 下面的2个是仅用原始的question 作为训练 测试集
# 6. self-consistency: entropy, max_ratio
# 7. selfCheckGPT: Contradiction in bert_score + n-gram neglog_probs + NLI inter_contradict
def merge_features(logits_file, group_file, label_file, json_suffix, out_suffix, 
    m_g_to_entropy_self=None, m_g_to_max_ratio_self=None, 
    m_g_to_max_contration_bert_self=None, m_g_to_max_contration_inter_self=None):
    # 写特征文件
    out_stream = open(logits_file + out_suffix, 'w')
    m_g_to_entropy = {}
    m_g_to_max_ratio = {}
    g_datas = read_json_sample_line(group_file)
    for g_data in g_datas:
        key = g_data['group_key']
        entropy, max_ratio = cal_entropy(g_data['responses'])
        m_g_to_entropy[key] = entropy
        m_g_to_max_ratio[key] = max_ratio

    m_instance_to_label = {}
    l_datas = read_json_sample_line(label_file)
    for l_data in l_datas:
        label = l_data['rule_label']
        m_label = int(l_data['model_label'])
        if label == -1:
            label = m_label
        
        m_instance_to_label[l_data['input']] = label

    logits_datas = read_json_sample_line(logits_file)
    my_rank = 0
    for logits_data in logits_datas:
        gd_answers = logits_data['original_ouput'] if isinstance(logits_data['original_ouput'], list) else [logits_data['original_ouput']]
        if gd_answers[0] == "" or gd_answers[0].find('en.wikipedia.org') >= 0:
            continue

        entropy = m_g_to_entropy[logits_data['Definition']]
        group_key = logits_data['Definition']
        max_ratio = m_g_to_max_ratio[logits_data['Definition']] 
        label = m_instance_to_label[logits_data['input']]
        token_probs, _, _ = cal_token_probs(logits_data['token_logits'])
        perplexity = get_perplexity(logits_data['token_logits'])
        neglog_probs, _ = cal_neglog_probs(logits_data['token_logits'])

        if m_g_to_entropy_self is None or m_g_to_max_ratio_self is None:
            text_arr = [logits_data['atypicality'], entropy, token_probs, logits_data['atypicality_avg'], 
                max_ratio, perplexity, neglog_probs, label]
        else:
            text_arr = [logits_data['atypicality'], entropy, token_probs, logits_data['atypicality_avg'], 
                max_ratio, perplexity, neglog_probs, 
                m_g_to_max_contration_bert_self[group_key], m_g_to_max_contration_inter_self[group_key],
                m_g_to_entropy_self[group_key], m_g_to_max_ratio_self[group_key],
                label]

        out_stream.write(','.join(map(str, text_arr)) + '\n')
        out_stream.flush()
        my_rank += 1
    print ('[samples]', my_rank)

def get_self_features(group_file):
    # 写特征文件
    m_g_to_entropy = {}
    m_g_to_max_ratio = {}
    m_g_to_max_contration_bert = {}
    m_g_to_max_contration_inter = {}
    m_g_to_input = {}
    g_datas = read_json_sample_line(group_file)
    for g_data in g_datas:
        key = g_data['group_key']
        entropy, max_ratio = cal_entropy(g_data['responses'])
        m_g_to_entropy[key] = entropy
        m_g_to_max_ratio[key] = max_ratio
        contra_bert_score = cal_bert_contradict_score(g_data['responses'])
        inter_contradict = cal_inter_contradict(g_data['responses'])
        m_g_to_max_contration_bert[key] = contra_bert_score
        m_g_to_max_contration_inter[key] = inter_contradict

        m_g_to_input[key] = g_data['instructions'][0][0]
    return m_g_to_entropy, m_g_to_max_ratio, m_g_to_max_contration_bert, m_g_to_max_contration_inter

def merge_features_self(logits_file, group_file, label_file, json_suffix, out_suffix):
    # 写特征文件
    print ('[begin parse features]')
    out_stream = open(logits_file + out_suffix, 'w')
    m_g_to_entropy = {}
    m_g_to_max_ratio = {}
    m_g_to_max_contration_bert = {}
    m_g_to_max_contration_inter = {}
    m_g_to_input = {}
    g_datas = read_json_sample_line(group_file)
    for g_data in g_datas:
        key = g_data['group_key']
        entropy, max_ratio = cal_entropy(g_data['responses'])
        m_g_to_entropy[key] = entropy
        m_g_to_max_ratio[key] = max_ratio

        contra_bert_score = cal_bert_contradict_score(g_data['responses'])
        inter_contradict = cal_inter_contradict(g_data['responses'])
        m_g_to_max_contration_bert[key] = contra_bert_score
        m_g_to_max_contration_inter[key] = inter_contradict

        m_g_to_input[key] = g_data['instructions'][0][0]

    # 获取label
    m_instance_to_label = {}
    l_datas = read_json_sample_line(label_file)
    for l_data in l_datas:
        label = l_data['rule_label']
        m_label = int(l_data['model_label'])
        if label == -1:
            label = m_label
        m_instance_to_label[l_data['input']] = label

    logits_datas = read_json_sample_line(logits_file)
    used_keys = set()
    my_rank = 0
    for logits_data in logits_datas:
        # 这里只存放原始的question
        # if m_g_to_input[key] != logits_data['input']:
        #     continue
        if logits_data['Definition'] in used_keys:
            continue
        used_keys.add(logits_data['Definition'])

        gd_answers = logits_data['original_ouput'] if isinstance(logits_data['original_ouput'], list) else [logits_data['original_ouput']]
        if gd_answers[0] == "" or gd_answers[0].find('en.wikipedia.org') >= 0:
            continue

        entropy = m_g_to_entropy[logits_data['Definition']]
        max_ratio = m_g_to_max_ratio[logits_data['Definition']] 
        label = m_instance_to_label[logits_data['input']]
        token_probs, _, _ = cal_token_probs(logits_data['token_logits'])
        perplexity = get_perplexity(logits_data['token_logits'])
        neglog_probs, _ = cal_neglog_probs(logits_data['token_logits'])

        contra_bert_score = m_g_to_max_contration_bert[key]
        inter_contradict = m_g_to_max_contration_inter[key]

        # TODO debug code
        # print ('[input]', logits_data['input'])
        # print ('[contra_bert_score, inter_contradict]', contra_bert_score, inter_contradict)
        # print ('[entropy, max_ratio, neglog_probs ]', entropy, max_ratio, neglog_probs )
        # print ('\n')

        text_arr = [logits_data['atypicality'], token_probs, entropy, max_ratio, 
            contra_bert_score, inter_contradict, neglog_probs, label]
        out_stream.write(','.join(map(str, text_arr)) + '\n')
        out_stream.flush()
        my_rank += 1
    print ('[samples]', my_rank)


    return False

def process_feature_file(used_file):
    # used_file = filename_arc
    print ('[process file]', used_file)
    # used_model = 'vicuna' # vicuna, llama2
    for used_model in ['llama2']:
        # vicuna
        logits_file = used_file + '.in.gen_%s_13B.index-0_10000' % (used_model)
        label_file = used_file +'.in.gen_%s_13B.index-0_10000.label_v2.index-0_10000' % (used_model)
        group_file = logits_file + '.before_group.grouped.index-0_1000'
        
        self_logits_file = used_file + '.in_ori.gen_%s_13B.index-0_10000' % (used_model)
        self_group_file = self_logits_file + '.before_group.grouped.index-0_1000'

        json_suffix = '.features.json'
        text_suffix = '.features.mergeall'
        # 基于logits_file 计算特征，并增加后缀写入logits_file
        print ('[process features %s]' % (used_model))
        m_g_to_entropy, m_g_to_max_ratio, m_g_to_max_contration_bert, m_g_to_max_contration_inter = get_self_features(self_group_file)
        # merge_features(logits_file, group_file, label_file, json_suffix, text_suffix)
        merge_features(logits_file, group_file, label_file, json_suffix, text_suffix,
            m_g_to_entropy, m_g_to_max_ratio, m_g_to_max_contration_bert, m_g_to_max_contration_inter)
        print ('[process self features]')
        # merge_features_self(self_logits_file, self_group_file, label_file, json_suffix, text_suffix)

def get_file_name(used_file, used_model):
    logits_file = used_file + '.in.gen_%s_13B.index-0_10000' % (used_model)
    label_file = used_file +'.in.gen_%s_13B.index-0_10000.label_v2.index-0_10000' % (used_model)
    group_file = logits_file + '.before_group.grouped.index-0_1000'
    
    self_logits_file = used_file + '.in_ori.gen_%s_13B.index-0_10000' % (used_model)
    self_group_file = self_logits_file + '.before_group.grouped.index-0_1000'

    return logits_file, label_file, group_file

def report_sampls_atypicality(label_file, logits_file):
    m_instance_to_label = {}
    l_datas = read_json_sample_line(label_file)
    for l_data in l_datas:
        label = l_data['rule_label']
        m_label = int(l_data['model_label'])
        if label == -1:
            label = m_label
        m_instance_to_label[l_data['input']] = label

    logits_datas = read_json_sample_line(logits_file)
    my_rank = 0
    NUM_LABEL = 2
    atypicality_list = [ [] for i in range(NUM_LABEL)]
    atypicality_avg_list = [ [] for i in range(NUM_LABEL)]
    used_keys = set()
    group_statistics = {}
    group_to_label_cnt = {}
    group_to_label = {}
    for logits_data in logits_datas:
        gp_key = logits_data['Definition']
        # if logits_data['Definition'] in used_keys:
        #     continue
        # used_keys.add(logits_data['Definition'])
        
        label = m_instance_to_label[logits_data['input']]
        # atypicality_list[label].append(logits_data['atypicality'])
        # atypicality_avg_list[label].append(logits_data['atypicality_avg'])

        if gp_key in group_statistics:
            group_statistics[gp_key].append(logits_data['atypicality'])
            group_to_label[gp_key].append(label)
        else:
            group_statistics[gp_key] = [logits_data['atypicality']]
            group_to_label[gp_key] = [label]

    for gp_key, gp_value in group_to_label.items():
        has_neg = False
        has_pos = False
        # print ('', gp_value)
        for item in gp_value:
            if item == 0:
                has_neg = True
            elif item == 1:
                has_pos = True
        if has_neg and has_pos:
            group_to_label_cnt[gp_key] = 1

    sts = [0.0, 0.0]
    for gp_key, gp_value in group_to_label_cnt.items():
        labels = group_to_label[gp_key]
        statistics = group_statistics[gp_key] 
        atypicality_list = [ [] for i in range(NUM_LABEL)]
        for k, v in enumerate(labels):
            atypicality_list[v].append(statistics[k])
        for i in range(NUM_LABEL):
            sts[i] += sum(atypicality_list[i])/ len(atypicality_list[i])
            # print ('%d\t%f' % (i, sum(atypicality_list[i])/ len(atypicality_list[i])))
    print ('[sts]', sts)
    

    print ('[keys comp]', len(group_to_label.keys()), len(group_to_label_cnt.keys()))
    # report
    # print ('[atypicality]')
    sum_val = 0
    for i in range(NUM_LABEL):
        sum_val += len(atypicality_list[i])
    for i in range(NUM_LABEL):
        print ('%d\t%d\t%f\t%f' % (i, len(atypicality_list[i]), len(atypicality_list[i]) * 1.0 /sum_val, sum(atypicality_list[i])/ len(atypicality_list[i])))
    
    # print ('[atypicality AVG]')
    # for i in range(NUM_LABEL):
    #     print ('%d\t%d\t%f' % (i, len(atypicality_avg_list[i]), sum(atypicality_avg_list[i])/ len(atypicality_avg_list[i])))


if __name__=="__main__":
    main_f = './datasets_self_detect/'

    filename_comqa = main_f + 'comqa/comqa_dev.json.multi_quest_agree'  #前300条验证qq simialr，剩余900条
    filename_faviq = main_f + 'FaVIQ/faviq_a_set_v1.2/train.jsonl.shuffle.multi_quest_agree'
    filename_truthfulQA = main_f + 'TruthfulQA/TruthfulQA.csv.shuffle.multi_quest_agree'
    
    ## process数学题目
    filename_gsm_8k = main_f + 'gsm_8k/test_shuffle.jsonl.multi_quest_agree'
    filename_svamp = main_f + 'SVAMP/data/mawps-asdiv-a_svamp/dev_shuffle.csv.multi_quest_agree' 
    
    ## process commonsense reasoning task
    filename_arc = main_f + 'AI2_ARC/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl.multi_quest_agree' 
    filename_arc = main_f + 'AI2_ARC/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl.multi_quest_agree' 
    commonsenseQA = main_f + 'commonsenseQA/dev_rand_split_shuffle.jsonl.multi_quest_agree'

    # used_file = filename_arc

    #for used_file in [filename_arc, commonsenseQA, filename_gsm_8k, filename_svamp, filename_faviq, filename_comqa]:
    for used_file in [filename_gsm_8k, filename_svamp]:
    
        # used_model = 'vicuna' # vicuna, llama2
        for used_model in ['vicuna']:
            print ('[task %s, model %s]' % (used_file, used_model))
            logits_file, label_file, group_file = get_file_name(used_file, used_model)
            report_sampls_atypicality(label_file, logits_file)
            print ('')

    # process_feature_file(commonsenseQA)
    # process_feature_file(filename_arc)

    # process_feature_file(filename_gsm_8k)
    # process_feature_file(filename_svamp)

    # process_feature_file(filename_comqa)
    # process_feature_file(filename_faviq)
    # 0       2187
    # 1       2823