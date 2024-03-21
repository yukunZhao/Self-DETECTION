# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import argparse
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence, List

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from peft import LoraConfig
from peft.utils.config import TaskType


from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
import random

# zhaoyukun02
import json
import requests
import os
import sys
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch

from transformers import pipeline

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template, add_model_args
import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

os.environ["WANDB_API_KEY"] = "c29c5de23235eb4762ad8b2d5a31c87182824c49"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
os.environ["debug"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # TODO zhaoyukun02
    conv = get_conversation_template("../vicuna-7b-v1.3/")
    # conv 返回的是Conversation 对象
    #conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def llama_config():
    ## seq_length = prompt_length + max_new_tokens
    return TRLConfig(
        train=TrainConfig(
            seq_length=768,
            epochs=1,
            total_steps=5000000,
            batch_size=1,
            checkpoint_interval=20000,
            eval_interval=20000,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=False,
        ),
        # peft_config=dict(peft_type="LORA")
        # model_path= ../vicuna-7b-v1.3, ./model/vicuna_SFT_7wtrain
        model=ModelConfig(model_path="../vicuna-7b-v1.3", num_layers_unfrozen=2, peft_type="LORA", peft_config=dict(peft_type="LORA")),
        tokenizer=TokenizerConfig(tokenizer_path="../vicuna-7b-v1.3", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-5)),
        
        # 原始的 num_rollouts 128, chunk_size 128
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=32, 
            chunk_size=1,
            ppo_epochs=4,
            init_kl_coef=0.001, # 一般设置0.1，值越大penalty越大，限制和init_model的近似度。防止reward hacking
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=128,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )

def build_prompt(instructions, input):
    return instructions[0] + '\n' + input

def generate_for_prompt(args, msgs):
    outputs = []
    for msg in msgs:
        conv = get_conversation_template(args.model_name_or_path)
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        gen_params = {
            "model": args.model_name_or_path,
            "prompt": prompt,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        response = requests.post(
            args.worker_address + "/worker_generate_stream",
            headers={"User-Agent": "FastChat Client"},
            json=gen_params,
            stream=False,
        )
        output = ""
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                output = data["text"].strip()
        #print ('[input, output]', ":", msg, ':', output)
        outputs.append(output)
    return outputs

def read_samples(TOP_N_samples=500):
    main_folder = './natural-instructions-2.8/'
    task_split_folder = main_folder + 'splits/default/'
    task_folder = main_folder + 'tasks/'
    train_tasks = task_split_folder + 'mysplit_train_tasks_700.txt'
    eval_tasks = task_split_folder + 'mysplit_eval_tasks_56.txt'
    USED_TASK = train_tasks #eval_tasks, train_tasks

    # read paraphrased instructions
    rephrase_task = USED_TASK + '.instruct_rephrases'
    m_task_to_rephrase_instructions = {}
    for line in open(rephrase_task):
        data = json.loads(line.strip())
        m_task_to_rephrase_instructions[data['Task_id']] = data['Definition_reprs']

    # read the instruction file, and output 100 samples for each task
    inputs = []
    # train_tasks, eval_tasks
    for line in open(USED_TASK):
        filename = task_folder + line.strip() + '.json'
        with open(filename, 'r') as f:
            dataset = json.load(f)
        sample = {}
        instruction_task = dataset['Definition'] # list[string]
        cates = dataset['Categories']
        paraphrases_all = []
        if line.strip() not in m_task_to_rephrase_instructions:
            continue
        paraphrases_tasks = m_task_to_rephrase_instructions[line.strip()]
        # 获取样本
        instances = dataset['Instances'] # input string, output list[string]
        random.shuffle(instances)
        instances_sample = instances[:TOP_N_samples]

        # 获取 同一个任务 N个Instruct * 100个input output
        instruct_all = instruction_task.copy()
        instruct_all.extend(paraphrases_tasks)
        for instruct in instruct_all:
            input_instance = [{"prompt": instruct + '\n' + instance['input'], "original_output": instance['output']} for instance in instances_sample ]
            inputs.extend(input_instance)
    return inputs

def build_verify_prompt(question, answer, answer_ref):
    prompt_template = """Evaluate the correctness of answer 'A' for question 'Q' based on the reference answer 'A_reference', and give a correct level from 0 to 4. Level 0 is a lack of understanding of the expected answer, and the provided answer type is different with reference answer being incorrect. Level 1 is a lack of understanding of the expected answer, and the answer 'A' happens to describe the same thing with the reference answer. Level 2 is an understanding of the expected answer, but the answer is incorrect. Level 3 is a full comprehension of the expected answer for question 'Q', with the answer 'A' almostly being correct but with minor errors.  Level 4 is a complete understanding of the expected answer for question 'Q', and the answer is entirely correct. 

Q: "We would like you to assess the QUALITY of each of the following argument (discussing Gay Marriage) and determine if the argument is Valid or Invalid. A valid argument is clearly interpretable and either expresses an argument, or a premise or a conclusion that can be used in an argument for the topic of gay marriage. An invalid argument is a phrase that cannot be interpreted as an argument or not on the topic of gay marriage.\nBasically, Foundit's list of \"suspect posters\" seems to amounts to anyone who actually debates with him and disagrees with him any more than an nominal amount."
A: "Sure, I can help you assess the quality of arguments related to gay marriage. Please provide me with the arguments you would like me to evaluate.""
A_reference: "Invalid"
Result: Level 0. Explanation: The answer type is not the same as the desired answer type (not consistent with the refrence answer type), and the answer is inconsistent with the reference answer.

Q: "A sentence and a query are given to you. Your responsibility is to provide the correct answer based on the provided sentence.\nSentence: Linda and Melinda decided to kayak across a lake. Linda made it across the lake before Melinda did. Question: Which one of them didn't go as fast as the other?"
A: "Based on the sentence, it is clear that Linda made it across the lake before Melinda did. Therefore, the answer to the query is that Linda did not go as fast as the other."
A_reference: "Melinda"
Result: Level 2. Explanation: A's answer type is consistent with the expected reference answer (have the same answer type), but the answer is not consistent with A_reference. 

Q: "In this task you will be given a list of numbers and you should remove all duplicates in the list. If every number is repeated in the list an empty list should be returned. Your list should be numbers inside brackets, just like the given list.\n[5, 7, 5, 3, 4, 1, 2, 7]"
A: "To remove duplicates from a list of numbers inside brackets, you can use a combination of a for loop and a set data structure. Here's an example implementation in Python:\n```python\ndef remove_duplicates(numbers):\n    numbers = [int(num) for num in numbers]  # convert to integers\n    return list(set(numbers))\n\n# example usage\nnumbers = [5, 7, 5, 3, 4, 1, 2, 7]\nresult = remove_duplicates(numbers)\nprint(result)  # output: [5, 7, 3, 4, 1, 2, 7]\n```\nThe `remove_duplicates` function takes a list of numbers as input, converts each number to an integer using the `int` function, and then uses a set to remove duplicates. The resulting list is then returned.\n\nIn the example usage, the input list `numbers` contains duplicates, so the output list `result` contains only the unique numbers."
A_reference: [3, 4, 1, 2]
Result: Level 2. Explanation: A try to solve the question without understanding the ultimate task, and the answer is not cnosistent with the reference answer.

Q: "Your mission involves working with a sentence to identify instances where two consecutive words have their orders flipped. Determine if the sentence remains grammatically correct if the order of these two words is switched. Label the instances as \"Inversion\" or \"Original\" based on your judgment.\nThen he slowly turned the to astonished onlookers."
A: "In this sentence, the order of the words \"then\" and \"he\" should be switched for the sentence to remain grammatically correct. The corrected sentence would be: \"He then slowly turned to the astonished onlookers.\" Therefore, the instances of inversion in this sentence are:  \n * \"Then\" and \"he\"\n * \"slowly turned\" and \"slowly turned to\"\n * \"to the astonished onlookers\" and \"to the astonished onlookers.\""
A_reference: "Inversion"
Result: Level 1. Explanation: A's answer type is not consistent with the expected reference answer type (have the same answer type), but the answer happens describe the same thing with A_reference. 

Q: "To accomplish this task, compose an SQL query that includes the following steps: Select data from a table, specify the columns to be returned, choose the tables to be selected, rename the tables if necessary, join the tables together using the \"JOIN\" statement, limit the number of rows returned, use the \"ON\" statement to specify conditions for the rows to be returned, use the \"WHERE\" statement to specify conditions for the columns, group the rows using the \"GROUP BY\" statement, use the \"HAVING\" statement to specify conditions for the groups, return the columns from the grouped rows using aggregate functions, sort the data using the \"ORDER BY\" statement, and return a specific number of rows using the \"LIMIT\" statement.\nDid M3 star a British Italian screenwriter"
A: "I'm sorry, but I cannot determine whether M3 is a British Italian screenwriter based on the information provided. Could you please provide more context or clarify your question?"
A_reference: "SELECT count(*) WHERE {\n?x0 a ns:film.writer .\n?x0 ns:people.person.nationality ns:m.03rjj .\n?x0 ns:people.person.nationality ns:m.07ssc .\nM3 ns:film.film.starring/ns:film.performance.actor ?x0\n}"
Result: Level 0. Explanation: The bot aborts and fails to answer the question. And the answer type is not consistent with the desired answer type.

Q: "Your role in this task is to analyze the sentiment of English language book product reviews and categorize them as either \"POS\" for positive sentiment or \"NEG\" for negative sentiment based on their overall tone.\nHow to behave and contribute in this KNOWLEDGE BASE WORLD. . This book is first published in 1999.  Right in the beginning, Drucker defined management as NOT only business management, but management in all kinds of organization: government, university, hospital, army, non-profit organization, etc.  The book is easy to read, just below 200 pages.  The scope and content is much wider than how we think management usually is.  It describes the CERTAINTIES: collapsing birthrate, shift in performance, global competitiveness, etc, and of course, how it affects all kinds of organizations.  It talks about information challenges, the knowledge worker and how one manage oneself in this new century.  Concepts like \"individuals will outlive their organization\", \"how to defines one's goal and contribution\" are all interesting topics and issues that everyone have to face, since, this is the REALITIES.  I highly recommend this book to anyone, not just CEO/executives, but those want to know more about oneself, how to behave and contribute in this KNOWLEDGE BASE WORLD"
A: "Based on the sentiment of the provided book review, I would categorize it as \"POS\" for positive sentiment. The review highlights the book's scope and content, which is much wider than how we think management usually is, and describes the concepts that everyone has to face in the realities of the knowledge-based world. The reviewer also recommends the book to anyone, not just CEO/executives, which further emphasizes the positive sentiment."
A_reference: "POS"
Result: Level 4. Explanation: The answer is consistent with the desired (reference) answer, and the answer type is the same as the reference answer providing some explanations for the answer.

Q: "In this assignment, replace female or gender-neutral pronouns with male pronouns. If there are no pronouns, add a male-gendered word to the sentence.\nThanks, I find it relaxing. I like crosswords, it's good for my balding head."
A: "I find it relaxing. I like crosswords, it's good for my balding head."
A_refrence: "Thanks, I find it relaxing. Like crosswords, it's good for the brain."
Result: Level 3. Explanation: A's answer type is consistent with the expected reference answer A_reference, and the answers are almostly same with the reference answer but with minor erros.

Q: "Examine the given utterance and the recent conversation to determine if the uv-part strategy is being used, which is a selfish negotiation tactic employed to undermine the opponent's requirements. \nContext: 'Yes. Can you help me to give some extra fodd?' 'I am willing to give you the good, though it would help us--we have a large group.  I really would like to take the water.  We are camping out in a primitive campsite with no running water.' 'Ya sure. '\nUtterance: 'Sorry for the type--I meant 'food' not 'good'.\ud83d\ude42'"
A: "Based on the given utterance and the recent conversation, it does not appear that the UV-part strategy is being used. In the conversation, the speaker is requesting extra food, but they are also considering the needs of the group as a whole. They mention that they have a large group and that there is no running water at their campsite, which suggests that they are thinking about the needs of the group as a whole. Additionally, the speaker apologizes for their mistake and clarifies their request for food, which suggests that they are not using the UV-part strategy to undermine the opponent's requirements."
A_refrence: "No"
Result: Level 4. Explanation: A's answer type is consistent with the expected reference answer A_reference, and the answer describe the same thing with reference answer.
    """
    prompt = prompt_template + '\nQ: ' + question + '\nA:' + answer + '\nA_reference: ' + answer_ref + '\nResult:'
    return prompt

def build_verfication_prompt(question, answer, answer_ref, is_verify_type=False):

    verfiy_type_template_short = """Assess whether the answer type of 'A' is expected answer type for question 'Q', which is inferred from whether it is the same answer type as the reference answer 'A_reference'. If the answer has the same type as the reference answer, give "Expected answer type", otherwise give "Unexpected answer type." 
    """
    verfiy_type_template = """Assess whether the answer type of 'A' is expected answer type for question 'Q', which is inferred from whether it is the same answer type as the reference answer 'A_reference'. If the answer has the same type as the reference answer, give "Expected answer type", otherwise give "Unexpected answer type." 

Q: "Your mission involves working with a sentence to identify instances where two consecutive words have their orders flipped. Determine if the sentence remains grammatically correct if the order of these two words is switched. Label the instances as \"Inversion\" or \"Original\" based on your judgment.\nThen he slowly turned the to astonished onlookers."
A: "In this sentence, the order of the words \"then\" and \"he\" should be switched for the sentence to remain grammatically correct. The corrected sentence would be: \"He then slowly turned to the astonished onlookers.\" Therefore, the instances of inversion in this sentence are:  \n * \"Then\" and \"he\"\n * \"slowly turned\" and \"slowly turned to\"\n * \"to the astonished onlookers\" and \"to the astonished onlookers.\""
A_reference: "Inversion"
Result: "Unexpected answer type, Explanation: A fails to provide a conclusion whether the orders are flipped"

Q: "In this task you will be given a list of numbers and you should remove all duplicates in the list. If every number is repeated in the list an empty list should be returned. Your list should be numbers inside brackets, just like the given list.\n[5, 7, 5, 3, 4, 1, 2, 7]"
A: "To remove duplicates from a list of numbers inside brackets, you can use a combination of a for loop and a set data structure. Here's an example implementation in Python:\n```python\ndef remove_duplicates(numbers):\n    numbers = [int(num) for num in numbers]  # convert to integers\n    return list(set(numbers))\n\n# example usage\nnumbers = [5, 7, 5, 3, 4, 1, 2, 7]\nresult = remove_duplicates(numbers)\nprint(result)  # output: [5, 7, 3, 4, 1, 2, 7]\n```\nThe `remove_duplicates` function takes a list of numbers as input, converts each number to an integer using the `int` function, and then uses a set to remove duplicates. The resulting list is then returned.\n\nIn the example usage, the input list `numbers` contains duplicates, so the output list `result` contains only the unique numbers."
A_reference: [3, 4, 1, 2]
Result: "Expected answer type, Explanation: A contains the same list type as A_reference."

Q: "To accomplish this task, compose an SQL query that includes the following steps: Select data from a table, specify the columns to be returned, choose the tables to be selected, rename the tables if necessary, join the tables together using the \"JOIN\" statement, limit the number of rows returned, use the \"ON\" statement to specify conditions for the rows to be returned, use the \"WHERE\" statement to specify conditions for the columns, group the rows using the \"GROUP BY\" statement, use the \"HAVING\" statement to specify conditions for the groups, return the columns from the grouped rows using aggregate functions, sort the data using the \"ORDER BY\" statement, and return a specific number of rows using the \"LIMIT\" statement.\nDid M3 star a British Italian screenwriter"
A: "I'm sorry, but I cannot determine whether M3 is a British Italian screenwriter based on the information provided. Could you please provide more context or clarify your question?"
A_reference: "SELECT count(*) WHERE {\n?x0 a ns:film.writer .\n?x0 ns:people.person.nationality ns:m.03rjj .\n?x0 ns:people.person.nationality ns:m.07ssc .\nM3 ns:film.film.starring/ns:film.performance.actor ?x0\n}"
Result: "Unexpected answer type, Explanation: The bot aborts and fails to provide a SQL query like the reference answer."

Q: "Generate an appropriate title for the given text. The generated title must be short and include the main topic of the text. The preferred titles are under fifteen words.\nThe 12-day event drew to a close on Sunday after screening 134 new features. There were more than 50,000 admissions at venues across the city - an increase on last year's figure. The festival sold tickets for theatres including the EIFF's home at the Filmhouse. It also sold tickets at its venue partner Cineworld, and at the Festival Theatre Edinburgh, the Traverse, Dominion Cinema, Odeon Lothian Road, Royal Lyceum Theatre and Belmont Filmhouse in Aberdeen. Organisers said an additional 14,600 people enjoyed two weekends of outdoor screenings as part of the Film in the City event. Opening night Actor Robert Carlyle's directorial debut The Legend of Barney Thomson opened the 2015 festival - in its 69th year - where it received its world premiere ahead of its release at the end of July. Ken Hay, EIFF chief executive said there had been some \"truly memorable moments\", including the opening night and Ewan McGregor's sold-out in-person event at the Lyceum. \"Our audiences have obviously responded to our rich and diverse programme, and we are thrilled to see the increase in admissions,\" he said. As the event closed, it was announced Scottish actors James Cosmo and Karen Gillan had been made EIFF honorary patrons. Cosmo has appeared in films such as Highlander and Braveheart while Gillan is known for her role as Amy Pond in Doctor Who."
A: "Edinburgh International Film Festival Concludes with Increased Admissions"
A_reference: "Edinburgh International Film Festival admissions up by 9%"
Result: "Expected answer type, Explanation: The two answers are both generated titles about the text."

Q: "In this task, you will be presented with a text, a pronoun from the text, and two candidate names. You should determine what the pronoun refers to and classify the answers into A, B, or Neither. A and B here are referring to option A and option B. Position of the pronoun in the text is showed within two \"_\"s.\nDeb knows everything about Bani and transforms her into a strong, confident woman hellbent on getting revenge. Bani changes her name to Durga and lives with Deb's family. Deb is an equally angry and vengeful man, who has lost the love of his life, Pallavi, and believes Jai is responsible. One day, Jai sees Durga and instantly knows _she_ is Bani. <sep>, Pronoun: she <sep>, A: Deb <sep>, B: Pallavi"
A: "She refers to Durga (Bani who has changed her name to Durga)\nA: Deb (who has transformed Bani into a strong, confident woman hellbent on getting revenge)\nB: Pallavi (who Deb believes is responsible for taking away the love of his life)"
A_reference: "Neither"
Result: "Unexpected answer type, Explanation: The answer fails to provide what the pronoun refers like the reference answer."
    """
    prompt_correctness_template_short = """Assess whether the answer 'A' is correct and give "Incorrect," "Partially correct," or "Completely correct" compared to the reference answer. "Incorrect" indicates the answer is not the same as the reference answer and is not a reasonable answer for the question 'Q'. "Partially correct" indicates the answer is reasonable but with minor errors or not completely the same as the reference answer. "Completely correct" indicates the answer is reasonable for open questions or identical to the reference answer.
    """
    prompt_correctness_template = """Assess whether the answer 'A' is correct and give "Incorrect," "Partially correct," or "Completely correct" compared to the reference answer. "Incorrect" indicates the answer is not the same as the reference answer and is not a reasonable answer for the question 'Q'. "Partially correct" indicates the answer is reasonable but with minor errors or not completely the same as the reference answer. "Completely correct" indicates the answer is reasonable for open questions or identical to the reference answer.

Q: "To accomplish this task, compose an SQL query that includes the following steps: Select data from a table, specify the columns to be returned, choose the tables to be selected, rename the tables if necessary, join the tables together using the \"JOIN\" statement, limit the number of rows returned, use the \"ON\" statement to specify conditions for the rows to be returned, use the \"WHERE\" statement to specify conditions for the columns, group the rows using the \"GROUP BY\" statement, use the \"HAVING\" statement to specify conditions for the groups, return the columns from the grouped rows using aggregate functions, sort the data using the \"ORDER BY\" statement, and return a specific number of rows using the \"LIMIT\" statement.\nDid M3 star a British Italian screenwriter"
A: "I'm sorry, but I cannot determine whether M3 is a British Italian screenwriter based on the information provided. Could you please provide more context or clarify your question?"
A_reference: "SELECT count(*) WHERE {\n?x0 a ns:film.writer .\n?x0 ns:people.person.nationality ns:m.03rjj .\n?x0 ns:people.person.nationality ns:m.07ssc .\nM3 ns:film.film.starring/ns:film.performance.actor ?x0\n}"
Result: "Incorrect, Explanation: A is not the same or similar like A_reference."

Q: "In this assignment, replace female or gender-neutral pronouns with male pronouns. If there are no pronouns, add a male-gendered word to the sentence.\nThanks, I find it relaxing. I like crosswords, it's good for my balding head."
A: "I find it relaxing. I like crosswords, it's good for my balding head."
A_reference: "Thanks, I find it relaxing. Like crosswords, it's good for the brain."
Result: "Partially correct, Explanation: the answer is almostly same with the reference answer but with minor erros absence of term 'Thanks', as this question need exactly match. "

Q: "In this particular task, you are given passages containing references to people, places, or things. Your responsibility is to formulate questions that evaluate one's understanding of these references. It is important to link pronouns or other mentions to the appropriate entities they refer to. Avoid questions that can be answered without comprehending the paragraph, and avoid questions with multiple answers or that do not link phrases to the same entity. Each answer should be a clear and unambiguous phrase from the paragraph.\nContext: The Review of Politics was founded in 1939 by Gurian, modeled after German Catholic journals. It quickly emerged as part of an international Catholic intellectual revival, offering an alternative vision to positivist philosophy. For 44 years, the Review was edited by Gurian, Matthew Fitzsimons, Frederick Crosson, and Thomas Stritch. Intellectual leaders included Gurian, Jacques Maritain, Frank O'Malley, Leo Richard Ward, F. A. Hermens, and John U. Nef. It became a major forum for political ideas and modern political concerns, especially from a Catholic and scholastic tradition." 
A: "1. Who founded The Review of Politics in 1939?\n2. What was the inspiration for The Review of Politics?\n3. Who were the editors of The Review of Politics for 44 years?\n4. Who were some of the intellectual leaders of The Review of Politics?\n5. What was the main focus of The Review of Politics?"
A_reference: "Gurian created what in 1939 at Notre Dame?"
Result: "Completely correct, Explanation: the answer is reasonable for the question do not need exactly match for this type of question."

Q: "In this task you will be given a list of numbers and you should remove all duplicates in the list. If every number is repeated in the list an empty list should be returned. Your list should be numbers inside brackets, just like the given list.\n[5, 7, 5, 3, 4, 1, 2, 7]"
A: "To remove duplicates from a list of numbers inside brackets, you can use a combination of a for loop and a set data structure. Here's an example implementation in Python:\n```python\ndef remove_duplicates(numbers):\n    numbers = [int(num) for num in numbers]  # convert to integers\n    return list(set(numbers))\n\n# example usage\nnumbers = [5, 7, 5, 3, 4, 1, 2, 7]\nresult = remove_duplicates(numbers)\nprint(result)  # output: [5, 7, 3, 4, 1, 2, 7]\n```\nThe `remove_duplicates` function takes a list of numbers as input, converts each number to an integer using the `int` function, and then uses a set to remove duplicates. The resulting list is then returned.\n\nIn the example usage, the input list `numbers` contains duplicates, so the output list `result` contains only the unique numbers."
A_reference: [3, 4, 1, 2]
Result: "Incorrect, Explanation: the output in A is not the same as A_reference as it needs accurately answered for the question."

Q: "Examine the given utterance and the recent conversation to determine if the uv-part strategy is being used, which is a selfish negotiation tactic employed to undermine the opponent's requirements. \nContext: 'Yes. Can you help me to give some extra fodd?' 'I am willing to give you the good, though it would help us--we have a large group.  I really would like to take the water.  We are camping out in a primitive campsite with no running water.' 'Ya sure. '\nUtterance: 'Sorry for the type--I meant 'food' not 'good'.\ud83d\ude42'"
A: "Based on the given utterance and the recent conversation, it does not appear that the UV-part strategy is being used. In the conversation, the speaker is requesting extra food, but they are also considering the needs of the group as a whole. They mention that they have a large group and that there is no running water at their campsite, which suggests that they are thinking about the needs of the group as a whole. Additionally, the speaker apologizes for their mistake and clarifies their request for food, which suggests that they are not using the UV-part strategy to undermine the opponent's requirements."
A_refrence: "No"
Result: "Completely correct, Explanation: The answer describe the same thing with reference answer."

Q: "We would like you to assess the QUALITY of each of the following argument (discussing Gay Marriage) and determine if the argument is Valid or Invalid. A valid argument is clearly interpretable and either expresses an argument, or a premise or a conclusion that can be used in an argument for the topic of gay marriage. An invalid argument is a phrase that cannot be interpreted as an argument or not on the topic of gay marriage.\nBasically, Foundit's list of \"suspect posters\" seems to amounts to anyone who actually debates with him and disagrees with him any more than an nominal amount."
A: "Sure, I can help you assess the quality of arguments related to gay marriage. Please provide me with the arguments you would like me to evaluate.""
A_reference: "Invalid"
Result: "Incorrect, Explanation: do not give the argument Valid or Invalid and is diffrent from reference answer."
    """
    len_q = len(question.split(' '))
    len_a = len(answer.split(' '))
    len_af = len(answer_ref.split(' '))
    if is_verify_type:
        if len_q > 420 or len_q + len_a + len_af > 512:
             verfiy_type_template = verfiy_type_template_short
        prompt = verfiy_type_template + '\nQ: ' + question + '\nA:' + answer + '\nA_reference: ' + answer_ref + '\nResult:'
    else:
        if len_q > 420 or len_q + len_a + len_af > 512:
            prompt_correctness_template = prompt_correctness_template_short
        prompt = prompt_correctness_template + '\nQ: ' + question + '\nA:' + answer + '\nA_reference: ' + answer_ref + '\nResult:'
    return prompt

def get_reward_from_response_old(args, question, generated_answer, gd_answer):
    if generated_answer.strip() == gd_answer.strip():
        return 1.0
    if generated_answer.find(gd_answer) >= 0 or gd_answer.find(generated_answer) >=0:
        return 0.9

    prompt = build_verify_prompt(json.dumps(question), json.dumps(generated_answer), json.dumps(gd_answer)) 
    response = generate_for_prompt(args, [prompt])[0]
    # print ('[Rseponse]', response)
    def get_float_score_from_str(s):
        nums = []
        num_str = ""
        for i in s:
            if i.isnumeric():  # 判断字符是否为数字字符
                num_str += i
            elif num_str:
                nums.append(int(num_str))
                num_str = ""
        if num_str:  # 处理最后一个数字串
            nums.append(int(num_str))
        if len(nums) > 0:
            return nums[0]
        return -1

    number = -1
    number = get_float_score_from_str(response)
    if response.find('Level ' + str(number)) < 0 or number < 0 or number > 4:
        number = -1
    if number == 0:
        return 0.0
    elif number == 1 or number == 2:
        return 0.2
    elif number == 3 or number == 4:
        return number / 4.0
    else:
        return random.randint(0,10) / 100.0

def get_reward_from_response(args, question, generated_answer, gd_answer):
    reward_level = -1
    if generated_answer.lower().strip() == gd_answer.lower().strip():
        reward_level = 6
    elif generated_answer.lower().find(gd_answer.lower()) >= 0 or gd_answer.lower().find(generated_answer.lower()) >=0:
        if abs(len(generated_answer) - len(gd_answer)) < 48 and len(gd_answer) > 1: 
            reward_level = 5
    else:
        prompt = build_verfication_prompt(json.dumps(question), json.dumps(generated_answer), json.dumps(gd_answer), True) 
        response = generate_for_prompt(args, [prompt])[0]
        correct_prompt = build_verfication_prompt(json.dumps(question), json.dumps(generated_answer), json.dumps(gd_answer), False) 
        correct_response = generate_for_prompt(args, [correct_prompt])[0]

        desired_answer_type = -1
        response = response.lower()
        if response.find("unexpected answer type") >= 0:
            desired_answer_type = 0
        elif response.find("expected answer type") >= 0 or response.find("valid") >= 0:
            desired_answer_type = 1

        desired_answer_correct = -1
        correct_response = correct_response.lower()
        if correct_response.find("incorrect") >= 0:
            desired_answer_correct = 0
        elif correct_response.find("completely correct") >= 0:
            desired_answer_correct = 2
        elif correct_response.find("correct") >= 0:
            desired_answer_correct = 1
        reward_level = -1

        if desired_answer_type == 0:
            if desired_answer_correct < 2:
                reward_level = 0
            else:
                reward_level = 1
        elif desired_answer_type == 1:
            if desired_answer_correct == 0:
                reward_level = 1 # 1
            elif desired_answer_correct == 1:
                reward_level = 3
            elif desired_answer_correct == 2:
                reward_level = 4
        else: # expected_answer_type 未识别
            if desired_answer_correct == 2:
                reward_level = 2
            elif desired_answer_correct == 1:
                reward_level = 1
            elif desired_answer_correct == 0:
                reward_level = 0
    # translate reward_level to reward_score
    reward_score = random.randint(0, 10) / 100.0
    if reward_level != -1:
        reward_score = max(1.0, reward_level / 4.0)
    return reward_score
    # return reward_level, desired_answer_type, response, desired_answer_correct, correct_response

def train():
    global local_rank
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--model_name_or_path", type=str, default="../vicuna-7b-v1.3/")
    parser.add_argument("--temperature", type=float, default=0.0) # 这里仅供NLI时候inf使用
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=51) # 这里仅供NLI时候inf使用
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    # parser.add_argument("--outputdir", type=str, default="./model/ppo-eval-data-200")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    # load datesets, 每个任务最多load 1000个instances
    N_SAMPLES_PER_TASK = 5
    inputs = read_samples(N_SAMPLES_PER_TASK)
    random.shuffle(inputs)
    # debug code

    prompts = []
    eval_prompts = []
    train_counts = max(len(inputs) - 50, 50)
    # 这里一定注意 train、eval 数据是分开的
    for k, v in enumerate(inputs):
        if k < train_counts:
            prompts.append(v)
        elif k < len(inputs):
            eval_prompts.append(v)
        else:
            break
    print ('[loading dataset, samples, train_count, eval_count]', len(inputs), len(prompts), len(eval_prompts))
    
    hparams = {}
    config = TRLConfig.update(llama_config().to_dict(), hparams)

    # config.model.peft_config = LoraConfig(
    #     r=8,
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    # )

    print ('---------prompts--------------')

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        str_outputs = kwargs['outputs']
        str_prompts = kwargs['prompts']
        original_outputs = kwargs['original_output']

        rewards = []
        for idx, item in enumerate(str_outputs):
            reward = 0.0
            rtmp = ''
            reward = get_reward_from_response(args, str_prompts[idx],  str_outputs[idx], original_outputs[idx][0])
            rewards.append(reward)
            #print ('[input-outputs]', ':', str_prompts[idx], ':', item, ':', original_outputs[idx], ':', rtmp)
        return rewards

    print ('[PPO train begin...]')
    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config,
    )

    print ('[training finished.]')
    #trainer.save_pretrained('./model/ppo-eval-data-500')
    #trainer.save_pretrained('./model/vicuna-ppo-train-data-small')
    trainer.save_pretrained('./model/vicuna_ppo_train_5_10_700-vicuna_base')
    #trainer.save_pretrained('./model/vicuna_ppo_train_5_10_700-vicuna_SFT')
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":

    train()
