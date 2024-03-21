import json
import random
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

def get_filename(task, used_model, use_self):
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

    if task == 'gsm_8k':
        used_dataset = filename_gsm_8k
    elif task == 'svamp':
        used_dataset = filename_svamp
    elif task == 'ARC':
        used_dataset = filename_arc 
    elif task == 'faviq':
        used_dataset = filename_faviq
    elif task == 'comqa':
        used_dataset = filename_comqa
    elif task == 'commonsenseQA':
        used_dataset = commonsenseQA
    else:
        used_dataset = None
    # used_model = 'llama2' # vicuna, llama2
    feature_file_suffix = '.features'
    feature_file_suffix = '.features.mergeall'
    logits_file = used_dataset + '.in.gen_%s_13B.index-0_10000' % (used_model)
    self_logits_file = used_dataset + '.in_ori.gen_%s_13B.index-0_10000' % (used_model)
    if use_self:
        return self_logits_file + feature_file_suffix
    return logits_file + feature_file_suffix
    
def get_features(use_self, feature_name, dataset):
    if use_self:
        if feature_name == "self_consistency":
            return dataset[:, 2:4]
        elif feature_name == "selfCheckGPT":
            return dataset[:, 4:7]
        else:
            return dataset[:, 0:7]
    else:
        if feature_name == "ALL":
            return dataset[:, 0:7]
        elif feature_name == "4_combination":
            return dataset[:, 0:4]
        elif feature_name == "atypicality":
            return dataset[:, 0:1]
        elif feature_name == "entropy":
            return dataset[:, 1:2]
        elif feature_name == "token_probs":
            return dataset[:, 2:3]
        elif feature_name == "perplexity":
            return dataset[:, 5:6]
        elif feature_name == "atypicality_avg":
            return dataset[:, 3:4]
        elif feature_name == "atypicalities":
            return dataset[:, [0,3]]
        elif feature_name == "ALL_merges":
            return dataset[:, 0:11]
        elif feature_name == "ALL_wo_selfGPT":
            return dataset[:, [0,1,2,3,4,5,9,10]]
        elif feature_name == "ALL_wo_selfcons":
            return dataset[:, 0:9]
        else:
            return dataset[:, 0:7]
# if __name__=="__main__":
def fit_xgb(task, used_model, use_self):
    # task = 'faviq' # ARC, commonsenseQA, svamp, gsm_8k, faviq, comqa
    # # task = 'commonsenseQA'
    # used_model = 'vicuna' # vicuna, llama2
    # use_self = False # False, True
    if use_self:
        f_names = ['self_consistency', 'selfCheckGPT', 'random', 'random']
        # f_names = ['random']
    else:
        f_names = ['ALL', '4_combination', 'atypicality', 'entropy', 'token_probs', 'perplexity', 'atypicality_avg']
        f_names = ['atypicalities']
        f_names = ['ALL_merges', 'ALL_wo_selfGPT', 'ALL_wo_selfcons']
    
    train_file = get_filename(task, used_model, use_self)
    """ schema = 0-4: ['atypicality',  entropy, token_probs, 'atypicality_avg', 
            4-7: max_ratio, perplexity, neglog_probs, label]
    """
    """ self schema = 0-4: [atypicality, token_probs, entropy, max_ratio, 
            4-7: contra_bert_score, inter_contradict, neglog_probs, label]
    """
    """ merge self schema: 0-4: ['atypicality',  entropy, token_probs, 'atypicality_avg', 
            4-6: max_ratio, perplexity, 
            6-9: neglog_probs, contradiction_bert, constradition_self,
            9-11: max_ratio, self_consistency_drawn, 
            11: label]
    """
    # load data
    dataset = loadtxt(train_file, delimiter=",")
    # split data into X and y
    
    for feature_name in f_names:
        print ('[Task %s, model %s, Use feature %s]' % (task, used_model, feature_name))
        X = get_features(use_self, feature_name, dataset)
        # Y = dataset[:, 7]
        Y = dataset[:, 11]
        seed = 7
        
        test_size = 0.25
        if feature_name == 'random':
            test_size = 0.9
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
        
        # fit model no training data
        print ('[start training]')

        # make predictions for test data
        if feature_name == 'random':
            predictions = [random.randint(0,1) for i in range(len(y_test))]
        else:
            model = XGBClassifier()
            model.fit(X_train, y_train)
            # make predictions for test data
            y_pred = model.predict(X_test)
            predictions = [value for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print ('[AUC, AUC-PR, Accuracy]')
        print ('%.2f' % (roc_auc_score(y_test, predictions) * 100))
        print ('%.2f' % (average_precision_score(y_test, predictions) * 100))
        print("%.2f" % (accuracy * 100.0))
        print ('')


if __name__=="__main__":
    task = 'faviq' # ARC, commonsenseQA, svamp, gsm_8k, faviq, comqa
    # task = 'commonsenseQA'
    used_model = 'llama2' # vicuna, llama2
    use_self = False # False, True

    tasks = ['ARC', 'commonsenseQA', 'svamp', 'gsm_8k', 'faviq', 'comqa',]
    # tasks = ['faviq', 'comqa', 'ARC', 'svamp', 'gsm_8k']
    # tasks = ['commonsenseQA']
    for task in tasks:
        fit_xgb(task, used_model, use_self)