export CUDA_VISIBLE_DEVICES=0,1

test_file="datasets_self_detect/commonsenseQA/dev_rand_split_shuffle.jsonl.multi_quest_agree.in.gen_vicuna_13B.index-0_10000"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_label.py \
    --model-path ../vicuna-13b-v1.5 --model-id vicuna_7b_sft_dpo \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
    --gen-suffix .label_v2 \
    --math-problem 0 \
    --input-file ${test_file}


test_file="datasets_self_detect/AI2_ARC/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl.multi_quest_agree.in.gen_vicuna_13B.index-0_10000"
echo $test_file
python3 fastchat/llm_judge/gen_model_answer_label.py \
    --model-path ../vicuna-13b-v1.5 --model-id vicuna_7b_sft_dpo \
    --question-begin 0 \
    --question-end 10000 \
    --num-gpus-total 2 \
    --gen-suffix .label_v2 \
    --math-problem 0 \
    --input-file ${test_file}
