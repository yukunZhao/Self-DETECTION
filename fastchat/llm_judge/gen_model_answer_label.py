"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm
import re

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template

def build_consistent_prompt(question, answer_ref, answer, is_verify_type=False):
    if isinstance(answer_ref, list):
        answer_ref = answer_ref[0]
    template_short = """Determine whether the answer 'A' is same or contradicted compared with the answer 'Golden A' for the question 'Q'. For choosing answers from several choices, you need to check whether the first answer is the exact same as the golden answer (like "True", "False" or other names, numbers, digits, or entities etc.) mentioned in the question 'Q'. For math problems, you need to check the final digital number in the statement is the same as the golden number. For question answering, you need to check the first answer describe the same thing as the golden one. If the two answers are exact same you give "Same", otherwise, you give "Contradicted" as the output. 
    """
    prompt = template_short + '\nQ: ' + question + '\nA: ' + answer + '\nGolden A: ' + answer_ref + '\nResult: '

    return prompt
def parge_consistent_content(is_result):
    if is_result.find('same') >= 0 or is_result.find('not contradicted') >= 0 or is_result.find('Same') >= 0:
        return True
    return False

# 用规则判断数学题 和 选择题 判断其答案是否正确，答案正确1、错误0、未知-1
def rule_define_answer_right(ref_answer, my_answer, question, isinteger=False):
    # 从选择题 题干中提取候选答案
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
    # 如果选择题 其他的选项和当前答案对上了，那么一定是错的
    for opsite_answer in opsite_answers:
        # print ('[check oppo]', my_answer, ':', opsite_answer)
        if verify_one_answer(opsite_answer, my_answer):
            return 0
    return -1



def generate_individual(args, qs, tokenizer, model, temperature, max_new_token):
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # prompt 会在前缀的基础上加一些说明, 哄一哄模型
    # prompt = qs
    input_ids = tokenizer([prompt]).input_ids

    if temperature < 1e-4:
        do_sample = False
    else:
        do_sample = True
    try:
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_token
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        output = tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )
        if conv.stop_str:
            output = output[: output.find(conv.stop_str)]
        output = output.strip()

        if conv.name == "xgen" and output.startswith("Assistant:"):
            output = output.replace("Assistant:", "", 1).strip()
    except RuntimeError as e:
        print("ERROR question ID: ", question[task_id_key])
        output = "ERROR"
    return output

def run_eval(args,
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    print ('[USE ray]', use_ray)
    print ('[length questions]', len(questions))

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers


    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(args,
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(args,
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory, 
    
):
    model, tokenizer = load_model(
        model_path,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    m_inst_input_to_instruction = {}
    m_inst_input_to_response = {}
    m_inst_input_to_gd_truth_response = {}
    m_inst_input_to_task_id = {}

    for sample_ind in questions:

        question = sample_ind['input']
        gd_answer = sample_ind['original_ouput']
        response = sample_ind['response_vicuna']
        task_id = sample_ind['Task_id']

        consistent_prompt = build_consistent_prompt(question, gd_answer, response)
        print ('[prompt]', consistent_prompt)
        consistent_result = generate_individual(args, consistent_prompt, tokenizer, model, args.temperature, max_new_token)
        print ('[reponse]', consistent_result)
        consistent_flag = parge_consistent_content(consistent_result)
        print ('[consistent_flag]', consistent_flag)
        print ('[is_math_problem]', args.math_problem, args.math_problem==1)

        rule_label = rule_define_answer_right(gd_answer, response, question, args.math_problem==1)

        data_json = {
            'input' : question,
            'Instruction' : sample_ind['Instruction'],
            'response_vicuna' : response,
            'original_ouput' : gd_answer,
            'model_label' : consistent_flag,
            'rule_label' : rule_label,
            'Task_id' : task_id
        }
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps(data_json) + "\n")



def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--gen-suffix", type=str, required=True)
    parser.add_argument("--math-problem", type=int, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="xx",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    #question_file = f"data/{args.bench_name}/question.jsonl"
    question_file = args.input_file
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = args.input_file + args.gen_suffix + '.index-' + str(args.question_begin) + '_' + str(args.question_end)
        #answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args,
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
    )

    #reorg_answer_file(answer_file)
