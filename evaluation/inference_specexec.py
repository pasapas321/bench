"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from evaluation.eval import run_eval, reorg_answer_file

from model.specexec.spec_generator import create_spec_generator
from model.specexec.specdec import utils
from model.specexec.specdec.trees import Tree

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

device = torch.device("cuda:0")

def specexec_forward(inputs, model, tokenizer, max_new_tokens, spec_generator=None,
                     seed=0, max_budget=16, max_beam_len=32, device=torch.device("cuda:0")):
    input_ids = inputs.input_ids
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    input_ids = input_ids.clone()
    accept_length_list = []
    
    spec_generator.prefix_tokens = input_ids[0]
    spec_generator.original_num_tokens = len(spec_generator.prefix_tokens)

    utils.set_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    if hasattr(spec_generator, "tree"):
        del spec_generator.tree
    spec_generator.reset_engines(prefix_len=len(spec_generator.prefix_tokens), max_budget=max_budget, max_new_tokens=max_new_tokens)

    spec_generator.tree = Tree(prefix_tokens=spec_generator.prefix_tokens, draft_engine=spec_generator.draft_engine, tokenizer=spec_generator.tokenizer)
    spec_generator.levels = spec_generator.get_tree_levels()  # in case the child class works with fixed trees

    # warmup:
    stats0 = spec_generator.grow_tree(prefix_tokens=spec_generator.prefix_tokens, max_budget=max_budget, max_beam_len=max_beam_len)
    stats1, warmup_tokens = spec_generator.validate_tree()
    torch.cuda.empty_cache()

    warmup_tokens_tensor = torch.tensor(warmup_tokens).to(device)
    spec_generator.prefix_tokens = torch.cat((spec_generator.prefix_tokens, warmup_tokens_tensor))

    # main generation cycle
    iter = 1
    eos_flag = False

    while len(spec_generator.prefix_tokens) < max_new_tokens + spec_generator.original_num_tokens + len(warmup_tokens) and not eos_flag:
        stats0 = spec_generator.grow_tree(prefix_tokens=spec_generator.prefix_tokens, max_budget=max_budget, max_beam_len=max_beam_len)
        stats1, fresh_tokens = spec_generator.validate_tree()
        torch.cuda.empty_cache()

        if spec_generator.tokenizer.eos_token_id in fresh_tokens:
            fresh_tokens = fresh_tokens[: fresh_tokens.index(spec_generator.tokenizer.eos_token_id)]
            eos_flag = True
             
        fresh_tokens_tensor = torch.tensor(fresh_tokens).to(device)
        spec_generator.prefix_tokens = torch.cat((spec_generator.prefix_tokens, fresh_tokens_tensor))

        iter += 1
        accept_length_list.append(len(fresh_tokens))
    
    new_token = len(spec_generator.prefix_tokens) - spec_generator.original_num_tokens - len(warmup_tokens)
    output_ids = spec_generator.prefix_tokens.unsqueeze(0)

    return output_ids, new_token, iter, accept_length_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--drafter-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
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
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
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
        "--tree-max-len", 
        help="max length of tree and engine cache, should fit prompt, generated and speculated tokens", 
        default=4096, 
        type=int
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--max-budget", 
        help="speculation token budget for fixed trees; CAN SWEEP", 
        default=16
    )
    parser.add_argument(
        "--max-beam-len", 
        help="max beam len; CAN SWEEP", 
        default=32
    )

    args = parser.parse_args()
    
    question_file = f"data/{args.bench_name}/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    #model_name_0 = 'JackFram/llama-68m'
    #model_name_1 = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    
    spec_generator = create_spec_generator(
        model_name_0=args.drafter_path,
        model_name_1=args.model_path,
        draft_engine_class='padded',
        gen_type='sx_base',
        offload=False,
        tree_max_len=args.tree_max_len
    )
    
    tokenizer = spec_generator.tokenizer
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=specexec_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        spec_generator=spec_generator,
        seed=args.seed,
        max_budget=args.max_budget,
        max_beam_len=args.max_beam_len,
        device=device
    )
    
    reorg_answer_file(answer_file)