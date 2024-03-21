import json
import random
import numpy as np
import re
# 根据 rephrased_files list * top_n 训练样本


def read_json_sample_line(filename):
    data = []
    for line in open(filename):
        data.append(json.loads(line.strip()))
    return data

def read_json_sample_all(filename):
    with open(filename, 'rb') as fp:
        data = fp.read()
        return json.loads(data)


import hashlib
def get_filemd5(filename):
    with open(filename, 'rb') as fp:
        data = fp.read()
    file_md5= hashlib.md5(data).hexdigest()
    print(file_md5)    

def merge_samples_arcoss_file(filelist, outfile):
    samples = []
    for file in filelist:
        datas = read_json_sample_all(file)
        samples.extend(datas)
    out_stream = open(outfile, 'w')
    print ('[samples]', len(samples))
    out_stream.write(json.dumps(samples, indent=1) + '\n' )
    out_stream.flush()

# 转换成固定格式， instruction, input, gd_output, ori_instruction, task_id
def trans_to_sample_file(filename, out_suffix, task_id, multi_choice=False, duplicate=False):
    datas = read_json_sample_line(filename)
    m_task_instruction = {
        'commonsenseQA': 'You are given a quesion along with 4 or 5 options. Choose the most logical answer from the given 4 or 5 options which can be used as the answer for the question. Output the choice and the word from the correct option.',
        'ARC': 'You are given a quesion along with 4 or 5 options. Choose the most logical answer from the given 4 or 5 options which can be used as the answer for the question. Output the choice and the word from the correct option.',
        'SVAMP' : "You are provided with an arithmetic question. Your task is to compute the solution and provide a correct answer.",
        'GSM8k' : "You are provided with an arithmetic question. Your task is to compute the solution and provide a correct answer.",
        'faviq': "A question is presented to you in this task, and your job is to write a potentially correct answer.",
        'truthfulQA': "A question is presented to you in this task, and your job is to write a potentially correct answer.",
        'comqa': "A question is presented to you in this task, and your job is to write a potentially correct answer.",
    }

    out_stream = open(filename + out_suffix, 'w')
    print ('[samples]', len(datas))
    for data in datas:
        ori_instruction = data['question_ori']
        questions = []
        for question in data['questions']:
            if len(question) > 0:
                questions.extend(question)
        if multi_choice:
            questions = np.array(data['questions']).flatten()
        questions_bak = questions
        # 把原始的问题重复10遍，后面的代码不用改
        if duplicate:
            questions = [questions_bak[0] for i in range(10)]

        for question in questions:      
            # for question in questions:
            sample = {
                'instruction' : m_task_instruction[task_id],
                'input' : question,
                'gd_output': data['gd_answers'],
                'ori_instruction' : ori_instruction,
                'ori_question' : ori_instruction,
                'task_id' : task_id
            }
            out_stream.write(json.dumps(sample) + '\n' )
            out_stream.flush()



def group_generations_by_instruction_input(input_file, out_suffix):
    out_stream = open(input_file + out_suffix, 'w')

    m_inst_input_to_instruction = {}
    m_inst_input_to_response = {}
    m_inst_input_to_gd_truth_response = {}
    m_inst_input_to_task_id = {}

    questions = read_json_sample_line(input_file)
    for sample_ind in questions:
        # print ('[sample_ind]', sample_ind.keys())
        key = sample_ind['Definition']
        if key not in m_inst_input_to_instruction:
            m_inst_input_to_instruction[key] = [sample_ind['input']]
            m_inst_input_to_response[key] = [sample_ind['response_vicuna']]
            m_inst_input_to_gd_truth_response[key] = sample_ind['original_ouput']
            m_inst_input_to_task_id[key] = sample_ind['Task_id']
        else:
            m_inst_input_to_instruction[key].append(sample_ind['input'])
            m_inst_input_to_response[key].append(sample_ind['response_vicuna'])

    group_cnt = 0
    instance_cnt = 0
    for group_key, generated_reponses in m_inst_input_to_response.items():
        group_cnt += 1
        # print ('group_key', group_key)
        # print ('generated_reponses', generated_reponses)
        instance_cnt += len(generated_reponses)
        data_json = {
            'group_key' : group_key,
            'instructions' : m_inst_input_to_instruction[group_key],
            'responses' : generated_reponses,
            'original_ouput' :  m_inst_input_to_gd_truth_response[group_key],
            'Task_id' : m_inst_input_to_task_id[group_key]
        }
        out_stream.write(json.dumps(data_json) + '\n')
    print ('[samples group_cnt, instance_cnt]', group_cnt, instance_cnt, instance_cnt / group_cnt)

def rule_define_answer_right(ref_answer, my_answer, question, isinteger=False):
    opsite_answers = []
    if question.find('(A)') >= 0 and question.find('(B)') >= 0 and question.find('(C)') >= 0:
        answers = question.split('\n')
        choices = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']
        for ans_tmp in answers:
            for choice in choices:
                if ans_tmp.find(choice) >= 0:
                    ans_tmp = ans_tmp.split(choice)[1].strip().strip('.').strip('?').strip('!').strip()
                    if ans_tmp.lower() != ref_answer.lower():
                        opsite_answers.append(ans_tmp)
                    break
    # 从选择题和数学题中提取答案
    pattern = r"(.*?) answer (.*?) is (.*\.?)"
    match_obj = re.match(pattern, my_answer)
    if match_obj:
        answer = match_obj.group(3)
        my_answer = answer.split('.')[0]
        # print (['match obj'], my_answer)
    else:
        pattern = r"(.*?) answer is (.*\.?)"
        match_obj = re.match(pattern, my_answer)
        if match_obj:
            answer = match_obj.group(2)
            my_answer = answer.split('.')[0]
            # print (['match obj 2nd'], my_answer)
    if isinteger:
        my_answer = my_answer.replace('\n', ' ')
        my_answer = my_answer.replace('\r', ' ')
        d_pattern = re.compile(r"\d+,?\d*|\d+\.?\d*")
        d_pattern2 = re.compile(r"\d+\.?\d*")
        numbers = d_pattern.findall(my_answer)
        numbers_2 = d_pattern2.findall(my_answer)
        if len(numbers_2) < len(numbers):
            numbers = numbers_2
        if len(numbers) == 1:
            my_answer = numbers[0].replace(',', '')
        else:
            pattern = r"(.*?) Answer\: \\boxed\{(.*?)\}"
            match_obj = re.search(pattern, my_answer)
            # print (['match obj 4nd'], my_answer)
            if match_obj:
                answer = match_obj.group(2)
                my_answer = answer.split('.')[0]
            else:
                pattern = r"(.*)Therefore(.*\.)"
                pattern2 = r"(.*)So(.*\.)"
                match_obj = re.search(pattern, my_answer)
                match_obj2 = re.search(pattern2, my_answer)
                if match_obj or match_obj2:
                    if match_obj:
                        my_answer = match_obj.group(2)
                    else:
                        my_answer = match_obj2.group(2)
                    # print (['match obj 4nd'], my_answer)
                    # 这里匹配 12,2 或者 12.34 这类数字
                    pattern = re.compile(r"\d+,?\d*|\d+\.?\d*")
                    pattern2 = re.compile(r"\d+\.?\d*")
                    new = pattern.findall(my_answer)
                    new2 = pattern2.findall(my_answer)
                    if len(new2) < len(new):
                        new = new2

                    if len(new) == 1:
                        my_answer = new[0].replace(',', '')
                    elif len(new) >0 and my_answer.find('=') >= 0:
                        my_answer = new[-1].replace(',', '')
                    
                    # print ('[match obj 5nd]', my_answer)

               

    def verify_one_answer(gd_answer, gen_answer):
        if isinstance(gd_answer, str):
            gd_answer = gd_answer.strip().strip('.').lower()
            gen_answer = gen_answer.strip().strip('.').lower()
            if gd_answer == gen_answer:
                return True
            # 这里如果是数字的话，不相等就算了
            if isinteger:
                return False
            len_gd = len(gd_answer)
            len_gen = len(gen_answer)
            if gen_answer.find(gd_answer) >= 0:
                if len_gen - len_gd < 5 or (len_gen < 1.5 * len_gd and (len_gen - len_gd) < 15 ):
                    return True
            return False 
        elif isinstance(gd_answer, list):
            for ans in gd_answer:
                if verify_one_answer(ans, gen_answer):
                    return True
            return False
        return False
    # print ('[check]', my_answer, ':', ref_answer)
    if verify_one_answer(ref_answer, my_answer):
        return 1
    elif isinteger:
        try:
            ref_answer_int = float(ref_answer)  
            my_answer_int = float(my_answer)
            if ref_answer_int != my_answer_int:
                return 0
        except:
            erroinfo = 1
    
    for opsite_answer in opsite_answers:
        # print ('[check oppo]', my_answer, ':', opsite_answer)
        if verify_one_answer(opsite_answer, my_answer):
            return 0
    return -1

def read_generations(input_file):
    questions = read_json_sample_line(input_file)
    random.shuffle(questions)
    for sample_ind in questions[:100]:

        question = sample_ind['input']
        gd_answer = sample_ind['original_ouput']
        response = sample_ind['response_vicuna']
        task_id = sample_ind['Task_id']

        # if response.find("Answer: ") <0:
        #     continue
        # print (question)
        print (gd_answer)
        print (json.dumps(response))
        print (rule_define_answer_right(gd_answer, response, question, True))
        print ('')

if __name__=="__main__":
    # parse_test_GPT4_generations_from_sun()
    # read_truthfulQA, read_comQA, read_quora_questions, read_faviq
    main_f = './datasets_self_detect/'
    out_suffix = '.in_ori'
    de_suffix = '.multi_quest_agree'

    filename_comqa = main_f + 'comqa/comqa_train.json.multi_quest_agree'  # 验证相同question产生不同答案
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
    
    # step 0, 把原始的 数据集文件转换成 需要的格式
    # trans_to_sample_file(commonsenseQA, out_suffix, 'commonsenseQA', True, True) # 5000+
    # trans_to_sample_file(filename_arc, out_suffix, 'ARC', True, True) # 2600
    # trans_to_sample_file(filename_svamp, out_suffix, 'SVAMP', False, True) # 5000
    # trans_to_sample_file(filename_gsm_8k, out_suffix, 'GSM8k', False, True) # 3876

    # trans_to_sample_file(filename_truthfulQA, out_suffix, 'truthfulQA',  False, True) # 939
    # trans_to_sample_file(filename_faviq, out_suffix, 'faviq',  False, True) # 3240
    # trans_to_sample_file(filename_comqa, out_suffix, 'comqa',  False, True) # 2916

    group_suffix = '.before_group'
    in_suffix = '.in.gen_llama2_13B.index-0_10000'
    in_suffix = '.in_ori.gen_vicuna_13B.index-0_10000'
    in_suffix = '.in_ori.gen_llama2_13B.index-0_10000'
    # group_generations_by_instruction_input(filename_gsm_8k + in_suffix, group_suffix)
    # group_generations_by_instruction_input(filename_svamp + in_suffix, group_suffix)
    # group_generations_by_instruction_input(filename_arc + in_suffix, group_suffix)
    # group_generations_by_instruction_input(commonsenseQA + in_suffix, group_suffix)

    group_generations_by_instruction_input(filename_truthfulQA + in_suffix, group_suffix)
    group_generations_by_instruction_input(filename_faviq + in_suffix, group_suffix)
    group_generations_by_instruction_input(filename_comqa + in_suffix, group_suffix)




   