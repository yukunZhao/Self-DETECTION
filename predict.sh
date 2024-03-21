export CUDA_VISIBLE_DEVICES=0,1
model="../vicuna-7b-v1.3/"
model="../vicuna-13b-v1.5/"
model="../llama-2-13b-chat_hf"

model_id="vicuna-13b-v1.5"
gen_suffix=".gen_vicuna_13B"

test_file=datasets_self_detect/commonsenseQA/dev_rand_split_shuffle.jsonl.multi_quest_agree.in
test_file=datasets_self_detect/commonsenseQA/dev_rand_split_shuffle.jsonl.multi_quest_agree.in_ori
python3 fastchat/llm_judge/gen_model_answer_with_probs.py \
    --model-path $model \
    --model-id $model_id \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
    --temperature 1.0 \
	--gen-suffix $gen_suffix \
    --input-file $test_file

exit 0

