export CUDA_VISIBLE_DEVICES=0,1

main_f="./datasets_self_detect/"
model="../llama-2-13b-chat_hf"
#model="../vicuna-13b-v1.5"

model_id="llama2-13b"
gen_suffix=".gen_llama2_13B"

test_file=${main_f}"AI2_ARC/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl.multi_quest_agree.in"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_with_probs_llama.py \
    --model-path $model \
    --model-id $model_id \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
	--gen-suffix $gen_suffix \
    --input-file $test_file


test_file=${main_f}"SVAMP/data/mawps-asdiv-a_svamp/dev_shuffle.csv.multi_quest_agree.in"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_with_probs.py \
    --model-path $model \
    --model-id $model_id \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
	--gen-suffix $gen_suffix \
    --input-file $test_file


test_file=${main_f}"gsm_8k/test_shuffle.jsonl.multi_quest_agree.in"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_with_probs.py \
    --model-path $model \
    --model-id $model_id \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
	--gen-suffix $gen_suffix \
    --input-file $test_file



test_file=${main_f}"comqa/comqa_dev.json.multi_quest_agree.in"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_with_probs.py \
    --model-path $model \
    --model-id $model_id \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
	--gen-suffix $gen_suffix \
    --input-file $test_file

test_file=${main_f}"FaVIQ/faviq_a_set_v1.2/train.jsonl.shuffle.multi_quest_agree.in"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_with_probs.py \
    --model-path $model \
    --model-id $model_id \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
	--gen-suffix $gen_suffix \
    --input-file $test_file

test_file=${main_f}"TruthfulQA/TruthfulQA.csv.shuffle.multi_quest_agree.in"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_with_probs.py \
    --model-path $model \
    --model-id $model_id \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
	--gen-suffix $gen_suffix \
    --input-file $test_file

test_file=datasets_self_detect/commonsenseQA/dev_rand_split_shuffle.jsonl.multi_quest_agree.in
python3 fastchat/llm_judge/gen_model_answer_with_probs.py \
    --model-path $model \
    --model-id $model_id \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
	--gen-suffix $gen_suffix \
    --input-file $test_file

exit 0

