import torch
import os
import requests
import json
import argparse
import random
import sys
from fastchat.model import load_model, get_conversation_template, add_model_args
from fastchat.model.model_adapter import get_conversation_template

# def generate_for_prompt_local(args, msgs, model, tokenizer):
#     outputs = []
#     for msg in msgs:
#         conv = get_conversation_template(args.model_name_or_path)
#         conv.append_message(conv.roles[0], msg)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()

#         inputs = tokenizer([prompt])
#         inputs = {k: torch.tensor(v).to(args.device) for k, v in inputs.items()}
#         output_ids = model.generate(
#             **inputs,
#             do_sample=True,
#             temperature=args.temperature,
#             repetition_penalty=args.repetition_penalty,
#             max_new_tokens=args.max_new_tokens,
#         )

#         if model.config.is_encoder_decoder:
#             output_ids = output_ids[0]
#         else:
#             output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
#         output = tokenizer.decode(
#             output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
#         )
#         outputs.append(output)

#     return outputs

def generate_for_prompt(args, msgs, specific_worker_address=None):
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
        if specific_worker_address is None:
            response = requests.post(
                args.worker_address + "/worker_generate_stream",
                headers={"User-Agent": "FastChat Client"},
                json=gen_params,
                stream=False,
            )
        else:
            response = requests.post(
                specific_worker_address + "/worker_generate_stream",
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

def build_prompt(prompt):
    prompt_template = """Paraphrase the input sentences with the same meaning but with different verbalizations, i.e., with different words and expressions, and output the several paraphrases seperated by '\n\n'. 
Input: In this task, you are given a piece of an article. Your task is to generate a short summary of the text. Try to give the summary in just one sentence.
Output: In this particular undertaking, you will be provided with a segment of an article. Your responsibility entails crafting a concise overview of the content. Endeavor to encapsulate the summary within a single sentence. \n\n Given an article excerpt, create a brief single-sentence summary of the content. \n\n Summarize the piece of the article, and try to give the summary as short as possible. \n\n Extracted from an article, create a concise summary in just one sentence. \n\n Your mission involves working with an article extract to craft a concise, one-sentence summary of its content.
Input: In this task, you're given the beginning and the ending of a three-part story. Your job is to complete the short story by writing a middle sentence that seamlessly connects  the first and last sentence. Generated sentences must be short, have fewer than 10 words, and be simple as if narrating to a child. Avoid using any irrelevant extra information when creating the middle sentence, and use names (e.g. Jack, Barbara,...) instead of pronouns (e.g. he / she) wherever possible.
Output: Formulate a concise, <10-word middle sentence to link the opening and closing of a 3-part story cohesively. \n\n Complete a 3-part story by composing a succinct, <10-word middle sentence to connect the first and last lines. \n\n Write a simple, <10-word sentence to link the beginning and ending of a 3-part story.
Input: In this task, you have given an input which is agent's reply, based on that you have to identify what will be users' command for that reply. 
Output: You are shown a random response (from a computer). Generate a language query such that it leads to this reply. \n\n You've been provided with an agent's response in this assignment. Your task involves genrating the corresponding user command based on the given reply. \n\n Generating a user query based on a given reply which is generated by a chating bot. 
Input: In this task, given a sentence in English Tokens, your task is to convert it into the English language. 
Output: Your objective for this assignment is to translate a sentence provided in English tokens into the English language. \n\n Convert the sentence in English Tokens to the English language. \n\n The aim of this task is to change a sentence written in English tokens into understandable English language. \n\n Convert a sentence in English tokens into the corresponding text in the English language.
Input: """
    template = """Paraphrase the input sentences with the same meaning but with different verbalizations, i.e., with different words and expressions, and output the several paraphrases seperated by '\n\n'. 
Input: In this task, you are given a piece of an article. Your task is to generate a short summary of the text. Try to give the summary in just one sentence.
Output: 
    1. In this particular undertaking, you will be provided with a segment of an article. Your responsibility entails crafting a concise overview of the content. Endeavor to encapsulate the summary within a single sentence. 
    2. In this particular undertaking, you will be provided with a segment of an article. Your responsibility entails crafting a concise overview of the content. Endeavor to encapsulate the summary within a single sentence. 
    3. Given an article excerpt, create a brief single-sentence summary of the content. 
    4. Summarize the piece of the article, and try to give the summary as short as possible. 
    5. Extracted from an article, create a concise summary in just one sentence. 
    6. Your mission involves working with an article extract to craft a concise, one-sentence summary of its content.
    7. Within this assignment, you'll be furnished with a segment of an article. Your mission is to produce a succinct overview of the content by crafting a brief summary encapsulated in a single sentence.
    8. Your role in this task is to work with a portion of an article. The challenge is to generate a succinct overview of the material by crafting a summary contained within a sole sentence.
    9. In this task, you'll be presented with a fragment of an article. Your goal is to formulate a concise summary of the content, aiming to present it in only one sentence.
    10. Generate a succinct overview of the piece of an article by crafting a summary contained within a sole sentence. The piece of an article is presented as following.
Input: """
    return template + prompt + '\nOutput: '


def parse_multi_paraphrases(outputs):
    results = []
    def get_nums(s):
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

    for output in outputs:
        output = output.strip().split('\n')
        for item in output:
            index = get_nums(item)
            item = item.strip().lstrip(str(index)).lstrip('\.').strip()
            if index > 0:
                results.append(item)
    return results

def vicuna_generate():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--model_name_or_path", type=str, default="../vicuna-7b-v1.3/")
    parser.add_argument("--temperature", type=float, default=0) # 这里仅供NLI时候inf使用
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024) # 这里仅供NLI时候inf使用
    parser.add_argument("--worker-address", type=str, default="http://localhost:21003") # 21003 vicuna; 21005 vicuna-ppo-10w
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    main_folder = './natural-instructions-2.8/'
    task_split_folder = main_folder + 'splits/default/'
    task_folder = main_folder + 'tasks/'
    train_tasks = task_split_folder + 'mysplit_train_tasks_700.txt'
    eval_tasks = task_split_folder + 'mysplit_eval_tasks_56.txt'
    test_tasks = task_split_folder + 'test_tasks.txt'

    paraphrases_tasks = test_tasks + '.instruct_rephrases'
    out = open(paraphrases_tasks, 'w')
    tasks = []
    start_index = 0
    myrank = 0
    print ('[start paraphrase]')
    for line in open(test_tasks):
        print (line.strip(), 'task begin ..')
        filename = task_folder + line.strip() + '.json'
        myrank += 1
        if myrank < start_index:
            continue
        with open(filename, 'r') as f:
            dataset = json.load(f)
        sample = {}
        instruction_task = dataset['Definition']
        cates = dataset['Categories']
        paraphrases_all = []

        print ('[instruction_task]', instruction_task)
        for instruction in instruction_task:
            prompt = build_prompt(instruction)
            generates = generate_for_prompt(args, [prompt])

            print ('[generates]', generates)
            paraphrases = parse_multi_paraphrases(generates)
            paraphrases_all.extend(paraphrases)
        
        sample['Task_id'] = line.strip()
        sample['Categories'] = dataset['Categories']
        sample['Definition'] = instruction_task
        sample['Definition_reprs'] = paraphrases_all
        sample['Samples'] = dataset['Positive Examples']

        if len(paraphrases_all) > 0:
            tasks.append(sample)
            out.write(json.dumps(sample))
            out.write('\n')
            out.flush()

        print (line.strip(), 'task done..')
    out.close()
    return -1


def read_samples():
    main_folder = './natural-instructions-2.8/'
    task_split_folder = main_folder + 'splits/default/'
    task_folder = main_folder + 'tasks/'
    # train_tasks = task_split_folder + 'mysplit_train_tasks_700.txt'
    eval_tasks = task_split_folder + 'mysplit_eval_tasks_56.txt'

    # read paraphrased instructions
    rephrase_task = eval_tasks + '.instruct_rephrases'
    m_task_to_rephrase_instructions = {}
    for line in open(rephrase_task):
        data = json.loads(line.strip())
        m_task_to_rephrase_instructions[data['Task_id']] = data['Definition_reprs']

    # read the instruction file, and output 100 samples for each task
    TOP_N_samples = 100
    inputs = []
    for line in open(eval_tasks):
        filename = task_folder + line.strip() + '.json'
        with open(filename, 'r') as f:
            dataset = json.load(f)
        sample = {}
        instruction_task = dataset['Definition'] # list[string]
        cates = dataset['Categories']
        paraphrases_all = []
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

Q: "A sentence and a query are given to you. Your responsibility is to provide the correct answer based on the provided sentence.\nSentence: Linda and Melinda decided to kayak across a lake. Linda made it across the lake before Melinda did. Question: Which one of them didn't go as fast as the other?"
A: "Based on the sentence, it is clear that Linda made it across the lake before Melinda did. Therefore, the answer to the query is that Linda did not go as fast as the other."
A_reference: "Melinda"
Result: Level 2. Explanation: A's answer type is consistent with the expected reference answer (have the same answer type), but the answer is not consistent with A_reference. 

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

Q: "In this task you will be given a list of numbers and you should remove all duplicates in the list. If every number is repeated in the list an empty list should be returned. Your list should be numbers inside brackets, just like the given list.\n[5, 7, 5, 3, 4, 1, 2, 7]"
A: "To remove duplicates from a list of numbers inside brackets, you can use a combination of a for loop and a set data structure. Here's an example implementation in Python:\n```python\ndef remove_duplicates(numbers):\n    numbers = [int(num) for num in numbers]  # convert to integers\n    return list(set(numbers))\n\n# example usage\nnumbers = [5, 7, 5, 3, 4, 1, 2, 7]\nresult = remove_duplicates(numbers)\nprint(result)  # output: [5, 7, 3, 4, 1, 2, 7]\n```\nThe `remove_duplicates` function takes a list of numbers as input, converts each number to an integer using the `int` function, and then uses a set to remove duplicates. The resulting list is then returned.\n\nIn the example usage, the input list `numbers` contains duplicates, so the output list `result` contains only the unique numbers."
A_reference: [3, 4, 1, 2]
Result: Level 2. Explanation: A try to solve the question without understanding the ultimate task, and the answer is not cnosistent with the reference answer.

Q: "Examine the given utterance and the recent conversation to determine if the uv-part strategy is being used, which is a selfish negotiation tactic employed to undermine the opponent's requirements. \nContext: 'Yes. Can you help me to give some extra fodd?' 'I am willing to give you the good, though it would help us--we have a large group.  I really would like to take the water.  We are camping out in a primitive campsite with no running water.' 'Ya sure. '\nUtterance: 'Sorry for the type--I meant 'food' not 'good'.\ud83d\ude42'"
A: "Based on the given utterance and the recent conversation, it does not appear that the UV-part strategy is being used. In the conversation, the speaker is requesting extra food, but they are also considering the needs of the group as a whole. They mention that they have a large group and that there is no running water at their campsite, which suggests that they are thinking about the needs of the group as a whole. Additionally, the speaker apologizes for their mistake and clarifies their request for food, which suggests that they are not using the UV-part strategy to undermine the opponent's requirements."
A_refrence: "No"
Result: Level 4. Explanation: A's answer type is consistent with the expected reference answer A_reference, and the answer describe the same thing with reference answer.

Q: "We would like you to assess the QUALITY of each of the following argument (discussing Gay Marriage) and determine if the argument is Valid or Invalid. A valid argument is clearly interpretable and either expresses an argument, or a premise or a conclusion that can be used in an argument for the topic of gay marriage. An invalid argument is a phrase that cannot be interpreted as an argument or not on the topic of gay marriage.\nBasically, Foundit's list of \"suspect posters\" seems to amounts to anyone who actually debates with him and disagrees with him any more than an nominal amount."
A: "Sure, I can help you assess the quality of arguments related to gay marriage. Please provide me with the arguments you would like me to evaluate.""
A_reference: "Invalid"
Result: Level 0. Explanation: The answer type is not the same as the desired answer type (not consistent with the refrence answer type), and the answer is inconsistent with the reference answer.
    """

    prompt_template_new = """Assess whether the answer 'A' is of the expected answer type for question 'Q', which can be inferred from whether it is the same answer type as the reference answer 'A_reference'. Determine whether the answer provides the same type as the reference answer, such as both answers being entities, numbers, options, statements, questions, titles, novels, codes, and so on, and give "Expected answer type" or "Unexpected answer type." In the second step, assess whether the answer is correct and give "Incorrect," "Partially correct," or "Completely correct" compared to the reference answer. "Incorrect" indicates the answer is not the same as the reference answer and is not a reasonable answer for open questions. "Partially correct" indicates the answer is reasonable with minor errors or not completely the same as the reference answer. "Completely correct" indicates the answer is reasonable for open questions or identical to the reference answer.

Q: "A sentence and a query are given to you. Your responsibility is to provide the correct answer based on the provided sentence.\nSentence: Linda and Melinda decided to kayak across a lake. Linda made it across the lake before Melinda did. Question: Which one of them didn't go as fast as the other?"
A: "Based on the sentence, it is clear that Linda made it across the lake before Melinda did. Therefore, the answer to the query is that Linda did not go as fast as the other."
A_reference: "Melinda"
Result: "Expected answer type" 
"Incorrect"

Q: "Your mission involves working with a sentence to identify instances where two consecutive words have their orders flipped. Determine if the sentence remains grammatically correct if the order of these two words is switched. Label the instances as \"Inversion\" or \"Original\" based on your judgment.\nThen he slowly turned the to astonished onlookers."
A: "In this sentence, the order of the words \"then\" and \"he\" should be switched for the sentence to remain grammatically correct. The corrected sentence would be: \"He then slowly turned to the astonished onlookers.\" Therefore, the instances of inversion in this sentence are:  \n * \"Then\" and \"he\"\n * \"slowly turned\" and \"slowly turned to\"\n * \"to the astonished onlookers\" and \"to the astonished onlookers.\""
A_reference: "Inversion"
Result: "Unexpected answer type"
"Partially correct"

Q: "To accomplish this task, compose an SQL query that includes the following steps: Select data from a table, specify the columns to be returned, choose the tables to be selected, rename the tables if necessary, join the tables together using the \"JOIN\" statement, limit the number of rows returned, use the \"ON\" statement to specify conditions for the rows to be returned, use the \"WHERE\" statement to specify conditions for the columns, group the rows using the \"GROUP BY\" statement, use the \"HAVING\" statement to specify conditions for the groups, return the columns from the grouped rows using aggregate functions, sort the data using the \"ORDER BY\" statement, and return a specific number of rows using the \"LIMIT\" statement.\nDid M3 star a British Italian screenwriter"
A: "I'm sorry, but I cannot determine whether M3 is a British Italian screenwriter based on the information provided. Could you please provide more context or clarify your question?"
A_reference: "SELECT count(*) WHERE {\n?x0 a ns:film.writer .\n?x0 ns:people.person.nationality ns:m.03rjj .\n?x0 ns:people.person.nationality ns:m.07ssc .\nM3 ns:film.film.starring/ns:film.performance.actor ?x0\n}"
Result: "Unexpected answer type", Explanation: The bot aborts and fails to answer the question. And the answer type is not consistent with the desired answer type.
"Incorrect"

Q: "In this assignment, replace female or gender-neutral pronouns with male pronouns. If there are no pronouns, add a male-gendered word to the sentence.\nThanks, I find it relaxing. I like crosswords, it's good for my balding head."
A: "I find it relaxing. I like crosswords, it's good for my balding head."
A_refrence: "Thanks, I find it relaxing. Like crosswords, it's good for the brain."
Result: "Expected answer type"
"Partially correct", Explanation: the answer is almostly same with the reference answer but with minor erros absence of term 'Thanks'. 

Q: "In this task you will be given a list of numbers and you should remove all duplicates in the list. If every number is repeated in the list an empty list should be returned. Your list should be numbers inside brackets, just like the given list.\n[5, 7, 5, 3, 4, 1, 2, 7]"
A: "To remove duplicates from a list of numbers inside brackets, you can use a combination of a for loop and a set data structure. Here's an example implementation in Python:\n```python\ndef remove_duplicates(numbers):\n    numbers = [int(num) for num in numbers]  # convert to integers\n    return list(set(numbers))\n\n# example usage\nnumbers = [5, 7, 5, 3, 4, 1, 2, 7]\nresult = remove_duplicates(numbers)\nprint(result)  # output: [5, 7, 3, 4, 1, 2, 7]\n```\nThe `remove_duplicates` function takes a list of numbers as input, converts each number to an integer using the `int` function, and then uses a set to remove duplicates. The resulting list is then returned.\n\nIn the example usage, the input list `numbers` contains duplicates, so the output list `result` contains only the unique numbers."
A_reference: [3, 4, 1, 2]
Result: "Expected answer type", Explanation: A try to solve the question without understanding the ultimate task, and the answer is not cnosistent with the reference answer.
"Incorrect"

Q: "Examine the given utterance and the recent conversation to determine if the uv-part strategy is being used, which is a selfish negotiation tactic employed to undermine the opponent's requirements. \nContext: 'Yes. Can you help me to give some extra fodd?' 'I am willing to give you the good, though it would help us--we have a large group.  I really would like to take the water.  We are camping out in a primitive campsite with no running water.' 'Ya sure. '\nUtterance: 'Sorry for the type--I meant 'food' not 'good'.\ud83d\ude42'"
A: "Based on the given utterance and the recent conversation, it does not appear that the UV-part strategy is being used. In the conversation, the speaker is requesting extra food, but they are also considering the needs of the group as a whole. They mention that they have a large group and that there is no running water at their campsite, which suggests that they are thinking about the needs of the group as a whole. Additionally, the speaker apologizes for their mistake and clarifies their request for food, which suggests that they are not using the UV-part strategy to undermine the opponent's requirements."
A_refrence: "No"
Result: "Expected answer type"  
"Completely correct", Explanation: The answer describe the same thing with reference answer.

Q: "We would like you to assess the QUALITY of each of the following argument (discussing Gay Marriage) and determine if the argument is Valid or Invalid. A valid argument is clearly interpretable and either expresses an argument, or a premise or a conclusion that can be used in an argument for the topic of gay marriage. An invalid argument is a phrase that cannot be interpreted as an argument or not on the topic of gay marriage.\nBasically, Foundit's list of \"suspect posters\" seems to amounts to anyone who actually debates with him and disagrees with him any more than an nominal amount."
A: "Sure, I can help you assess the quality of arguments related to gay marriage. Please provide me with the arguments you would like me to evaluate.""
A_reference: "Invalid"
Result: "Expected answer type"  
"Incorrect"
    """

    prompt = prompt_template_new + '\nQ: ' + question + '\nA:' + answer + '\nA_reference: ' + answer_ref + '\nResult:'
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
        if len_q + len_a + len_af > 768:
            prompt = verfiy_type_template + '\nQ: ' + question + '\nA:' + answer + '\nResult:'
    else:
        if len_q > 420 or len_q + len_a + len_af > 512:
            prompt_correctness_template = prompt_correctness_template_short
        prompt = prompt_correctness_template + '\nQ: ' + question + '\nA:' + answer + '\nA_reference: ' + answer_ref + '\nResult:'
        if len_q + len_a + len_af > 768:
            prompt = prompt_correctness_template + '\nQ: ' + question + '\nA:' + answer  + '\nResult:'
    return prompt

def get_reward_from_response(args, question, generated_answer, gd_answer):
    reward_level = -1
    if generated_answer.lower().strip() == gd_answer.lower().strip():
        reward_level = 6
    elif generated_answer.lower().find(gd_answer.lower()) >= 0 or gd_answer.lower().find(generated_answer.lower()) >=0:
        if abs(len(generated_answer) - len(gd_answer)) < 50 and len(gd_answer) > 1: 
            reward_level = 5
    else:
        prompt = build_verfication_prompt(json.dumps(question), json.dumps(generated_answer), json.dumps(gd_answer), True) 
        response = generate_for_prompt(args, [prompt], "http://localhost:21002")[0]
        correct_prompt = build_verfication_prompt(json.dumps(question), json.dumps(generated_answer), json.dumps(gd_answer), False) 
        correct_response = generate_for_prompt(args, [correct_prompt], "http://localhost:21002")[0]

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

        # print ('[verify answer_type]', desired_answer_type, response)
        # print ('[verify correctness]', desired_answer_correct, correct_response)

    reward_score = random.randint(0, 10) / 100.0
    if reward_level != -1:
        reward_score = max(1.0, reward_level / 4.0)

    return reward_level
    #return reward_level, desired_answer_type, response, desired_answer_correct, correct_response


def do_test_generate():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--model_name_or_path", type=str, default="../vicuna-7b-v1.3/")
    parser.add_argument("--temperature", type=float, default=0) # 这里仅供NLI时候inf使用
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024) # 这里仅供NLI时候inf使用
    parser.add_argument("--worker-address", type=str, default="http://localhost:21005") # 21002 vs 21005(train_3.5w_ppo)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    main_folder = './natural-instructions-2.8/'
    task_split_folder = main_folder + 'splits/default/'
    task_folder = main_folder + 'tasks/'
    train_tasks = task_split_folder + 'mysplit_train_tasks_700.txt'
    eval_tasks = task_split_folder + 'mysplit_eval_tasks_56.txt'
    test_tasks = task_split_folder + 'test_tasks.txt'
    paraphrases_tasks = train_tasks + '.instruct_rephrases'

    USED_TASK = test_tasks

    #output_file = USED_TASK + '.generation.vicuna.ppo.train3.5w'
    output_file = USED_TASK + '.generation.vicuna_SFT'
    output_file = USED_TASK + '.generation.vicuna_ppo_SFT'
    
    # output_file = eval_tasks + '.SFT_data.json'
    out = open(output_file, 'w')

    convs_all = []
    for line in open(USED_TASK):
        # step1, 获得有哪些task
        filename = task_folder + line.strip() + '.json'
        print (line.strip(), 'begins ..')

        with open(filename, 'r') as f:
            dataset = json.load(f)
        # step2 获得task具体的instruction和instances
        instruction_task = dataset['Definition']
        cates = dataset['Categories']
        instances = dataset['Instances']
        TOP_N_INS_PER_TASK = 20
        # 这里千万不能shuffle
        #random.shuffle(instances)
        instances = instances[:TOP_N_INS_PER_TASK]
        # paraphrases_all = []
        
        for instruction in instruction_task:
            for idx, instance in enumerate(instances):
                ask_prompt = instruction + '\n' + instance['input']
                original_ouput = instance['output'][0]
                # build SFT train dataset
                # conv = {}
                # conv["id"] = cates[0] + str(idx)
                # q_conv = {"from":"human", "value":ask_prompt}
                # r_conv = {"from": "gpt", "value":original_ouput}
                # conv["conversations"] = [q_conv, r_conv]
                # convs_all.append(conv)

                response = generate_for_prompt(args, [ask_prompt])[0]
                #reward = get_reward_from_response(args, ask_prompt, response, instance['output'][0])
                reward = -1
                
                sample = {}
                sample['Task_id'] = line.strip()
                sample['Categories'] = dataset['Categories']
                sample['Definition'] = instruction_task
                sample['Instruction'] = instruction
                sample['input'] = instance['input']
                sample['response_vicuna'] = response
                sample['reward_vicuna'] = reward
                sample['original_ouput'] = original_ouput

                out.write(json.dumps(sample))
                out.write('\n')
                out.flush()

        print (line.strip(), 'task done..', len(instances))
    # out.write(json.dumps(convs_all, indent=1))
    # # # out.write('\n')
    print ("ALL samples count", len(convs_all))
    out.close()
    return -1

def do_re_generate():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--model_name_or_path", type=str, default="../vicuna-7b-v1.3/")
    parser.add_argument("--temperature", type=float, default=0) # 这里仅供NLI时候inf使用
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32) # 这里仅供NLI时候inf使用
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    main_folder = './natural-instructions-2.8/'
    task_split_folder = main_folder + 'splits/default/'
    task_folder = main_folder + 'tasks/'
    train_tasks = task_split_folder + 'mysplit_train_tasks_700.txt'
    eval_tasks = task_split_folder + 'mysplit_eval_tasks_56.txt'
    test_tasks = task_split_folder + 'test_tasks.txt'
    paraphrases_tasks = train_tasks + '.instruct_rephrases'

    TOP_N_INS_PER_TASK = 100
    output_file = test_tasks + '.generation.vicuna.ppo.train3.relabel'
    generated_file = test_tasks + '.generation.vicuna.ppo.train3.5w'
    out = open(output_file, 'w')

    idx = 0
    for line in open(generated_file):
        idx += 1
        if idx % 5 != 0:
            continue
        try:
            sample = json.loads(line)
            # sample['Task_id'] = line.strip()
            # sample['Categories'] = dataset['Categories']
            # sample['Definition'] = instruction_task
            # sample['Instruction'] = instruction
            # sample['input'] = instance['input']
            # sample['response_vicuna'] = response
            # sample['reward_vicuna'] = reward
            # sample['original_ouput'] = instance['output']

            instruction = sample['Instruction']
            input = sample['input']
            reward = sample['reward_vicuna']
            ask_prompt = instruction + '\n' + input

            new_reward, typeright, typecontent, answerright, answercontent = get_reward_from_response(args, ask_prompt, sample['response_vicuna'], sample['original_ouput'])
            print ('[Task_id]', sample['Task_id'])
            print ('[ask_prompt]', json.dumps(ask_prompt))
            print ('[response]', json.dumps(sample['response_vicuna']))
            print ('[gd_response]', json.dumps(sample['original_ouput']))
            print ('[new_reward]', new_reward)
            print ('[new_reward_detail_type]', typeright, len(ask_prompt.split(' ')), json.dumps(typecontent))
            print ('[new_reward_detail_ans]', answerright, len(ask_prompt.split(' ')), json.dumps(answercontent))
            print ('\n')
        except:
            print ('Done')

    out.close()
    return -1


if __name__=="__main__":

    #vicuna_generate()
    #do_test_generate()
    do_re_generate()

    sys.exit(0)
    
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--model_name_or_path", type=str, default="../vicuna-7b-v1.3/")
    parser.add_argument("--temperature", type=float, default=0) # 这里仅供NLI时候inf使用
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024) # 这里仅供NLI时候inf使用
    parser.add_argument("--worker-address", type=str, default="http://localhost:21005")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1
    
    samples = read_samples()
    random.shuffle(samples)
    samples_new = samples[:30]

    for sample in samples_new:
        response = generate_for_prompt(args, [sample["prompt"]])[0]
        print ('[prompt]', json.dumps(sample["prompt"]))
        print ('[response]', json.dumps(response))
        print ('[ground-truth]', sample["original_output"][0])

        score = get_reward_from_response(args, sample["prompt"], response, sample["original_output"][0]) 
        print ('[final_score]', score)
        print ('\n')