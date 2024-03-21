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

def build_verfication_prompt(question, answer, answer_ref, is_verify_type=False):

    verfiy_type_template_short = """Determine whether the answer 'A' is expected answer type for question 'Q'. For classification tasks (like sentiment analysis, selecting choice, entailment inference and so on), you need to check whether the answer is the exact same as one of the expected Enums (like “True”, ”False”, “Positive“, ”Negative”, “A”, “B”, “C”, “D”, “Contradiction”, “Neutral” or “Entailment” etc.) as mentioned in the question ‘Q’. For free-form generation tasks (like title generation, question generation, data-to-text generation and so on), you need to check whether the answer is an expected generated title, question, data-to-text description or a summarization, etc., as the Question ‘Q’ required. You also need to compare with the ‘Golden A’ to determine whether the answer type aligns with the answer type of the ‘Golden A’. If the answer 'A' has the same type as the golden answer, give "Expected answer type", otherwise give "Unexpected answer type". Please note that you only need to determine whether it matches the required answer type, and do not need to verify whether the answer is correct.
    """
    prompt_correctness_template_short = """Determine whether the answer 'A' is 'Correct' or 'Incorrect' for question 'Q'. For classification tasks (the answer is limited to a finite set such as “True”, ”False”, “Positive”, ”Negative”, “A”, “B”, “C”, “D”, “Contradiction”, “Neutral”, “Entailment”), you need to check whether the answer is exactly match (equals) the golden answer ‘Golden A’. For free-form generation tasks (the answer is a free-form generation and not unique such as title, question, data-to-text description generation or a summarization), you need to check whether the answer describe the same thing as the golden answer, or the answer is fluent, plausible for the question ‘Q’. If the answer 'A' is correct, give "Correct", otherwise give "Incorrect" as the result.
    """

    few_shot_verify_type = """
Q: We would like you to assess the QUALITY of each of the following argument (discussing Gay Marriage) and determine if the argument is Valid or Invalid. A valid argument is clearly interpretable and either expresses an argument, or a premise or a conclusion that can be used in an argument for the topic of gay marriage. An invalid argument is a phrase that cannot be interpreted as an argument or not on the topic of gay marriage.\nBasically, Foundit's list of \"suspect posters\" seems to amounts to anyone who actually debates with him and disagrees with him any more than an nominal amount.
Golden A: Invalid
A: Sure, I can help you assess the quality of arguments related to gay marriage. Please provide me with the arguments you would like me to evaluate.
Result: Unexpected answer type. 
Explanation: The answer type is not the same as the desired answer type, which needs to be Valid or Invalid as the question 'Q' asked. It is not same with the answer type of Golden A.

Q: Your objective in this task is to generate a topic word by incorporating at least one word from the given fact. The topic word should also include a new word from a related concept. Aim to select a topic word with two or more words for greater effectiveness.\nFact: a solar panel converts sunlight into electricity.
Golden A: sunlight sun
A: Solar panel electricity
Result: Expected answer type
Explanation: The answer A describe a topic word as the question 'Q' asked, and it is the same answer type with the Golden A.
    """
    few_shot_answer_correctness = """
Q: In this task, you will work with a piece of text that contains a pronoun and two possible names. Your objective is to identify the referent of the pronoun and classify it as A, B, or Neither. The location of the pronoun in the text will be indicated by two underscores.
Golden A: B
A: B
Result: Correct
Explanation: The answer is the exactly same as the Golden answer A, so it is correct.

Q: Your objective in this task is to generate a topic word by incorporating at least one word from the given fact. The topic word should also include a new word from a related concept. Aim to select a topic word with two or more words for greater effectiveness.\nFact: a solar panel converts sunlight into electricity.
Golden A: sunlight sun
A: Solar panel electricity
Result: Correct
Explanation: The answer A describe a reasonable topic word as the question 'Q' asked, and it is similar with the Golden A. So it is correct.

Q: Go through the provided text and determine the correct pronoun for the specified name. The chosen pronoun should replace the symbol(_). The target name is enclosed within ** **. Select a pronoun amongst 'her', 'him', 'he', 'she', 'his' that correctly matches the casing as per its location in the text. \nAnother brother, Dr. Aswin W. Sastrowardoyo, is a physician who was formerly a guitarist and vocalist with the music group Chaseiro from 1979 to 1983, and a younger sister, Lisa Damayanti Sastrowardoyo (b. 1962). The actress Dian Sastrowardoyo is a niece of Ms. **Joesoef**. _ is married to Mr. Iwan V. Joesoef, a businessman, and has two sons Marwan Arie Joesoef (born 26/5/1976), Nurfahd S. Joesoef (born 4/3/1979) and one daughter Tiara R. Joesoef (born 5/7/1999)
Golden A: she
A: he
Result: Incorrect
Explanation: The answer A must be the same as the Golden answer A as the question 'Q' described, but it is not. So the result is Incorrect. 
    """

    len_q = len(question.split(' '))
    len_a = len(answer.split(' '))
    len_af = len(answer_ref.split(' '))
    if is_verify_type:
        prompt = verfiy_type_template_short + few_shot_verify_type +  '\nQ: ' + question + '\nGolden A:' +  answer_ref + '\nA: ' + answer + '\nResult:'
        if len_q + len_a + len_af > 1024:
            prompt = verfiy_type_template_short + few_shot_verify_type + '\nQ: ' + question + '\nA:' + answer + '\nResult:'
    else:
        prompt = prompt_correctness_template_short + few_shot_answer_correctness + '\nQ: ' + question + '\nGolden A:' + answer_ref + '\nA: ' + answer + '\nResult:'
        if len_q + len_a + len_af > 1024:
            prompt = prompt_correctness_template_short + few_shot_answer_correctness + '\nQ: ' + question + '\nA:' + answer  + '\nResult:'
    return prompt

def parse_reward_from_response(generated_answer, gd_answer, response, correct_response):
    reward_level = -1
    desired_answer_correct = -1
    desired_answer_type = -1
    if generated_answer.lower().strip() == gd_answer.lower().strip():
        reward_level = 2
        desired_answer_correct = 1
        desired_answer_type = 1
    else:
        response = response.lower()
        if response.find("unexpected answer type") >= 0:
            desired_answer_type = 0
        elif response.find("expected answer type") >= 0 or response.find("valid") >= 0:
            desired_answer_type = 1
        
        correct_response = correct_response.lower()
        if correct_response.find("incorrect") >= 0:
            desired_answer_correct = 0
        elif correct_response.find("correct") >= 0:
            desired_answer_correct = 1
        
        if desired_answer_type == 0:
            reward_level = 0
        elif desired_answer_type == 1:
            if desired_answer_correct == 0:
                reward_level = 1
            elif desired_answer_correct == 1:
                reward_level = 2
            else:
                reward_level = 1
    return reward_level, desired_answer_type, desired_answer_correct

def generate_individual(args, qs, tokenizer, model, temperature, max_new_token):
    conv = get_conversation_template(args.model_path)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt = qs

    # print ('[prompt]', prompt)
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
        #TODO 这里llama 不能用 
        if args.model_path.find("llama") >= 0:
            skip = 1
        else:
            if conv.stop_str:
                output = output[: output.find(conv.stop_str)]
        output = output.strip()

        if conv.name == "xgen" and output.startswith("Assistant:"):
            output = output.replace("Assistant:", "", 1).strip()
    except RuntimeError as e:
        print("ERROR question ID: ", qs)
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

    for question in tqdm(questions):
        temperature = args.temperature
        # 这里注意改回去 
        gd_output_key = "original_ouput"
        instruction_key = "Instruction"
        ori_instruction_key = 'Definition'
        task_id_key = "Task_id"
        generated_answer = question['response_vicuna']
        turns = []

        gd_output = question[gd_output_key]
        if len(gd_output) == 0:
            continue

        if isinstance(gd_output, list) and (isinstance(gd_output[0], int) or len(gd_output) > 3):
            gd_output = json.dumps(gd_output[0])
        elif isinstance(gd_output, list):
            gd_output = gd_output[0]

        # if isinstance(gd_output, list):
        #     gd_output = '[' + ', '.join(map(str, gd_output)) + ']'

        # debug code 这里改动，把output也放在input里了
        qs = question[instruction_key] + '\n' + question['input']

        type_prompt = build_verfication_prompt(json.dumps(qs), json.dumps(generated_answer), json.dumps(gd_output), True) 
        correct_prompt = build_verfication_prompt(json.dumps(qs), json.dumps(generated_answer), json.dumps(gd_output), False) 
        # print ('[max_new_token]', max_new_token)
        # max_new_token = 512
        type_output = generate_individual(args, type_prompt, tokenizer, model, args.temperature, max_new_token)
        correct_output = generate_individual(args, correct_prompt, tokenizer, model, args.temperature, max_new_token)
        # print ('[reward content type_output]', max_new_token, type_output)
        # print ('[reward content correct_output]', max_new_token, correct_output)
        reward_total, type_reward, correct_reward = parse_reward_from_response(generated_answer, gd_output, type_output, correct_output)
        # print ('[rewards]', reward_total, type_reward, correct_reward)
        # "ori_instruction" : question['ori_instruction'],
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "Task_id": question[task_id_key],
                "Instruction" : question[instruction_key],
                "input" : question["input"],
                "original_ouput" : question[gd_output_key],
                "response_vicuna": generated_answer,
                "reward_total" : reward_total,
                "type_score" : type_reward,
                "correct_score" : correct_reward,
                "Definition": question[ori_instruction_key]
            }
            fout.write(json.dumps(ans_json) + "\n")


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
