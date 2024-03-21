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
import torch.nn.functional as F

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template


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
    # debug load more questions for shuffle
    questions = load_questions(question_file, question_begin, question_end * 10)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)
    # DEBUG limit the number of questions to process
    questions = questions[:question_end]

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
    # negative loglikelihood of the input texts
    def get_batch_loglikelihood(texts):
        # TODO 这里 vicuna 要改回来
        # tokenized = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        # 特殊TODO, llama 不 padding
        tokenized = tokenizer(texts, return_tensors="pt").to("cuda")
        labels = tokenized.input_ids
        outputs = model(**tokenized, labels=labels)
        logits = outputs.logits.cpu()
        labels = labels.cpu()
        # print ('[labels in get_batch_loglikelihood]', labels.shape)
        # print ('[logits in get_batch_loglikelihood]', logits.shape)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels[shift_labels == tokenizer.pad_token_id] = -100
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").detach()
        ll_per_sample = -loss.view(shift_logits.shape[0], shift_logits.shape[1])
        nonpad_per_row = (shift_labels != -100).sum(dim=1)
        ll_mean = ll_per_sample.sum(dim=1)/nonpad_per_row
        
        ll_per_sample[(shift_labels == -100)] = 0
        ll_total = ll_per_sample.sum(dim=1)
        torch.cuda.empty_cache()
        #ll_mean.cpu().numpy()
        return float(ll_mean), float(ll_total)

    # 自行设计的从output_score中获取scores的函数，这里的score是进softmax之前的值，是未归一化的值
    # 最终采用build-in fuction: compute_transition_scores，可以通过设置 normalize_logits 获得归一化的值
    def get_generate_logprobs(output_ids, output_scores):
        print ('[output_ids shape]', output_ids.shape)
        all_tokens = output_ids.shape[1] # 这个是包含了 input + output所有的序列
        output_ids = output_ids[0].tolist()

        new_tokens = len(output_scores)
        print ('[output_ids detail]', output_ids[all_tokens - new_tokens:])

        print ('[output_scores.shape]', new_tokens,  output_scores[0].shape)
        for i in range(new_tokens):
            id = int(torch.argmax(output_scores[i][0]))
            print ('[id individual]', id, float(output_scores[i][0][id]))

    for question in tqdm(questions):
        temperature = args.temperature

        choices = []
        input_embeddings = []
        # 这里注意改回去 
        gd_output_key = "gd_output"
        # gd_output_key = "response_vicuna"
        instruction_key = "instruction"
        ori_instruction_key = 'ori_instruction'
        task_id_key = "task_id"
        # for i in range(len(question[gd_output_key])):
        output = ''
        turns = []

        gd_output = question[gd_output_key]
        if len(gd_output) == 0:
            continue

        if isinstance(gd_output, list):
            gd_output = '[' + ', '.join(map(str, gd_output)) + ']'

        # debug code 这里改动，把output也放在input里了
        qs = question[instruction_key] + '\n' + question['input']

        conv = get_conversation_template(args.model_path)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        # prompt 会在前缀的基础上加一些说明, 哄一哄模型
        prompt = conv.get_prompt()
        # prompt = qs
        # print ('[ori q]', qs)
        # print ('[prompt]', prompt)
        # if args.model_path.find("llama") >=0:
        #     prompt = qs

        input_ids = tokenizer([prompt]).input_ids

        if temperature < 1e-4:
            do_sample = False
        else:
            do_sample = True

        # get the prompt loglikelihood
        #print ('before calcu prompt loglikelihood')
        prompt_avg_logprob, prompt_total_logprob = get_batch_loglikelihood(qs)
        #print ('[prompt loglikelihood]', prompt_total_logprob, prompt_avg_logprob)

        # some models may error out when generating long outputs
        try:
            output_ids = model.generate(
                torch.as_tensor(input_ids).cuda(),
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_token,
                output_hidden_states=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True

            )
            # debug code
            # 应该是一个map结果的, map结构： odict_keys(['sequences', 'scores', 'hidden_states', 'attentions'])
            # sequences (`tf.Tensor` of shape `(batch_size, sequence_length)`):
            #   The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            #   if all batches finished early due to the `eos_token_id`.
            # scores `tuple(tf.Tensor)` : max_new_tokens * `(batch_size, config.vocab_size)`
            #     Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            #     at each generation step. Tuple of `tf.Tensor` with up to `max_new_tokens` elements (one element for each
            #     generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
            # attentions `tuple(tuple(tf.Tensor))`: # generated_length * # num_layers * (batch_size, num_heads, sequence_length, sequence_length)
            #     Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            #     `tf.Tensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
            # hidden_states `tuple(tuple(tf.Tensor))`:  # generated_length * # num_layers * (batch_size, generated_length, hidden_size)
            # 第一个位置 存放（input_ids * hidden_size的表示），后面每个位置都是一个单独的hidden表示
            #     Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            #     `tf.Tensor` of shape `(batch_size, generated_length, hidden_size)`.
            hidden_states = output_ids['hidden_states']
            #print ('[input_ids len]', len(input_ids), len(input_ids[0]))
            output_scores = output_ids['scores']
            #print ('[output_scores len]', len(output_scores), output_scores[0].shape, output_scores[-1].shape)
            # output_scores len 49（生成的序列长度） torch.Size([1, 32000]) torch.Size([1, 32000])
            # print ('output_scores details', output_scores[0][0][0:5])

            # # 第一个位置 存放（input_ids * hidden_size的表示），后面每个位置都是一个单独的hidden表示
            # print ('[hidden_states], len 0:', len(hidden_states), len(hidden_states[0]), hidden_states[0][-1].shape, hidden_states[-1][-1].shape)
            # # [hidden_states], len 0: 5(生成长度) 33(layers) torch.Size([1, 273, 4096]) (273是input_ids的长度) torch.Size([1, 1, 4096])
            # print ('[hidden_states], len 1:', hidden_states[1][-1].shape) # torch.Size([1, 1, 4096])

            output_ids = output_ids['sequences'] # sequences:  batch_size * num_return_sequences * sequence_length
            
            # get_generate_logprobs(output_ids, output_scores)
            # trans_scores = model.compute_transition_scores(output_ids, output_scores, normalize_logits=False)
            # print ('[trans_scores]', trans_scores)
            trans_scores = model.compute_transition_scores(output_ids, output_scores, normalize_logits=True)
            trans_scores = trans_scores[0].tolist()
            #print ('[trans_scores normed]', trans_scores[0].tolist())

            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]) :]
            output = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            # zhaoyukun02 model_path里面包含llama2的就不搞
            print ('[original output]', output)

            if args.model_path.find("llama") >= 0:
                skip = 1
            else:
                if conv.stop_str:
                    output = output[: output.find(conv.stop_str)]
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()
        except RuntimeError as e:
            print("ERROR question ID: ", question[task_id_key])
            output = "ERROR"

        turns.append(output)
        conv.messages[-1][-1] = output

        # print ('[output]', output, "\n")

        # Dump answers
        # "ori_instruction" : question['ori_instruction'],
        #                "token_logits" : trans_scores,
        # "atypicality" : prompt_total_logprob,
        #         "atypicality_avg" : prompt_avg_logprob,
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "Task_id": question[task_id_key],
                "Instruction" : question[instruction_key],
                "input" : question["input"],
                "original_ouput" : question[gd_output_key],
                "response_vicuna": output,
                "token_logits" : trans_scores,
                "atypicality" : prompt_total_logprob,
                "atypicality_avg" : prompt_avg_logprob,
                
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
