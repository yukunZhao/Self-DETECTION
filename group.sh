export CUDA_VISIBLE_DEVICES=0,1
model="../vicuna-13b-v1.5"

test_file="datasets_self_detect/commonsenseQA/dev_rand_split_shuffle.jsonl.multi_quest_agree.in.gen_llama2_13B.index-0_10000.before_group"
test_file="datasets_self_detect/commonsenseQA/dev_rand_split_shuffle.jsonl.multi_quest_agree.in_ori.gen_vicuna_13B.index-0_10000.before_group"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_cluster_real.py \
    --model-path $model --model-id vicuna \
    --question-begin 0 \
    --question-end 1000 \
    --num-gpus-total 2 \
    --gen-suffix .grouped \
    --input-file ${test_file}

test_file="datasets_self_detect/AI2_ARC/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl.multi_quest_agree.in.gen_llama2_13B.index-0_10000.before_group"
test_file="datasets_self_detect/AI2_ARC/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl.multi_quest_agree.in_ori.gen_vicuna_13B.index-0_10000.before_group"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_cluster_real.py \
    --model-path $model --model-id vicuna \
    --question-begin 0 \
    --question-end 1000 \
    --num-gpus-total 2 \
    --gen-suffix .grouped \
    --input-file ${test_file}

