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

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template

# old version
# def build_consistent_prompt(question, answer, answer_ref, is_verify_type=False):
#     template_short = """Determine whether the answer 'A' is same or contradicted compared with the answer ‘A Reference’ for the question 'Q'. For classification tasks (like sentiment analysis, selecting choice, entailment inference and so on), you need to check whether the two answers are the exact same (like “True”, ”False”, “Positive“, ”Negative”, “A”, “B”, “C”, “D”, “Contradiction”, “Neutral” or “Entailment” or other names, numbers, digits, or entities etc.) as mentioned in the question ‘Q’. If the two answers are exact same you give “same”, otherwise, you give “Contradicted” as the output. For free-form generation tasks (like title generation, question generation, data-to-text generation and so on), you need to check whether the answer ‘A’ is an expected generated title, question, data-to-text description or a summarization, etc., as the answer ‘A Reference’. If the two answers describe the same meaning you give “Same”, otherwise, you give “Contradicted” as the output.
#     """
#     prompt = template_short + '\nQ: ' + question + '\nA: ' + answer + '\nA Reference: ' + answer_ref + '\nResult: '

#     return prompt

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
        key = sample_ind['group_key']
        group_key = key

        m_inst_input_to_task_id[key] = sample_ind['Task_id']
        m_inst_input_to_instruction[key] = sample_ind['instructions']
        m_inst_input_to_response[key] = sample_ind['responses']
        m_inst_input_to_gd_truth_response[key] = sample_ind['original_ouput']

        # for group_key, generated_reponses in m_inst_input_to_response.items():
        generated_reponses = sample_ind['responses']
        grouped_reponses = []
        grouped_instructions = []
        grouped_reponses.append([generated_reponses[0]])
        grouped_instructions.append([m_inst_input_to_instruction[group_key][0]])
        # current_group_instruction = []
        for idx, rep in enumerate(generated_reponses):
            if idx == 0:
                continue
            contradiction_flag_all = True
            for idy in range(len(grouped_reponses)):
                # 从已有的cluster里面随机选一个判断是否一致
                rep_ref = grouped_reponses[idy][0]
                consistent_prompt = build_consistent_prompt(group_key, rep_ref, rep)
                print ('[prompt]', consistent_prompt)
                consistent_result = generate_individual(args, consistent_prompt, tokenizer, model, args.temperature, max_new_token)
                print ('[reponse]', consistent_result)
                consistent_flag = parge_consistent_content(consistent_result)
                print ('[consistent_flag]', consistent_flag)
                if consistent_flag:
                    grouped_reponses[idy].append(rep)
                    grouped_instructions[idy].append(m_inst_input_to_instruction[group_key][idx])
                    contradiction_flag_all = False
                    break
            # 新开一个cluster
            if contradiction_flag_all:
                grouped_reponses.append([rep])
                grouped_instructions.append([m_inst_input_to_instruction[group_key][idx]])

        data_json = {
            'group_key' : group_key,
            'instructions' : grouped_instructions,
            'responses' : grouped_reponses,
            'original_ouput' :  m_inst_input_to_gd_truth_response[group_key],
            'Task_id' : m_inst_input_to_task_id[group_key]
        }
        with open(os.path.expanduser(answer_file), "a") as fout:
            fout.write(json.dumps(data_json) + "\n")
        # os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        # with open(os.path.expanduser(answer_file), "a") as fout:
        #     ans_json = {
        #         "Task_id": question[task_id_key],
        #         "Instruction" : question[instruction_key],
        #         "input" : question["input"],
        #         "original_ouput" : question[gd_output_key],
        #         "response_vicuna": generated_answer,
        #         "reward_total" : reward_total,
        #         "type_score" : type_reward,
        #         "correct_score" : correct_reward,
        #         "Definition": question[ori_instruction_key]
        #     }
        #     fout.write(json.dumps(ans_json) + "\n")


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
