import argparse
import datetime
import json
import logging
import os
import socket
import subprocess
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm

from model.specexec.offloading.offload_model import load_gptq_offloaded_model, load_offloaded_model
from model.specexec.specdec import SpecExecBeams, SpecExecBase, SpecInfer, utils
import model.specexec.engine
from model.specexec.specdec.utils import colored

device = torch.device("cuda:0")
_DEFAULT_DEVICE_SIZE = 2
DISPLAY_WIDTH = 160
pd.set_option("display.width", DISPLAY_WIDTH)
pd.set_option("display.max_columns", 32)


def create_spec_generator(
    model_name_0,
    model_name_1,
    draft_engine_class,
    gen_type="SX",
    offload=False,
    device_size=_DEFAULT_DEVICE_SIZE,
    check_tokenizer=False,
):
    """Creates a SpecGenerator object for different generation types.

    This function loads draft and target pre-trained language models specified by their names
    and creates a SpecBase subclass object based on the provided generation type.
    It also handles several configuration options like device placement and tokenizer verification.

    Args:
        model_name_0 (str): Name of the draft model.
        model_name_1 (str): Name of the target model.
        gen_type (str, optional): Generation type. Defaults to "SX" (SpecExec).
            Valid options include:
                - "SpecExecBase", : SpecExec generator
                - "SI", "spec_infer", "specinfer": SpecInfer generator
        offload (bool, optional): Whether to offload model 1 using offloading library. Defaults to False.
        device_size (int, optional): Device size for offloading. Defaults to `_DEFAULT_DEVICE_SIZE`.
        check_tokenizer (bool, optional): Whether to verify if both models have the same tokenizer. Defaults to False.

    Returns:
        SpecGenerator: An instance of a SpecBase subclass object based on the provided parameters.

    Raises:
        ValueError: If an invalid `gen_type` is provided.
    """

    if len(model_name_0.split("::")) == 2:
        model_name_0, rev_0 = model_name_0.split("::")
    else:
        rev_0 = "main"  # default in `from_pretrained()`

    if len(model_name_1.split("::")) == 2:
        model_name_1, rev_1 = model_name_1.split("::")
    else:
        rev_1 = "main"  # default in `from_pretrained()`

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_0, legacy=False)

    if check_tokenizer:
        # verify that the two models have the same tokenizer
        tokenizer_1 = transformers.AutoTokenizer.from_pretrained(model_name_1, legacy=False)
        vv0 = tokenizer.get_vocab()
        vv1 = tokenizer_1.get_vocab()

        ignored_tokens = ["[PAD]"]  # disregard these tokens when comparing the cokonizers' vocabs
        assert set(vv0.keys()).difference(ignored_tokens) == set(vv1.keys()).difference(ignored_tokens)
        for k in set(vv0.keys()).difference(ignored_tokens):
            assert vv0[k] == vv1[k]
        del tokenizer_1, vv0, vv1

    #logger.info(f"Loading Model 0: `{model_name_0}`, {draft_engine_class=}")
    if draft_engine_class.lower() in ("es", "static", "enginestatic"):
        model_0 = transformers.AutoModelForCausalLM.from_pretrained(model_name_0, device_map=device, torch_dtype=torch.float16, revision=rev_0)
        draft_engine = engine.EngineStatic(model_0, max_len=args.tree_max_len)
    # elif draft_engine_class.lower() in ("esc", "staticcompiled", "enginestaticcompiled"):
    #     model_0 = transformers.AutoModelForCausalLM.from_pretrained(model_name_0, device_map=device, torch_dtype=torch.float16, revision=rev_0)
    #     draft_engine = engine.EngineStaticCompiled(model_0, max_len=args.tree_max_len)
    # elif draft_engine_class.lower() in ("ie", "inferenceengine"):
    #     draft_engine = engine.InferenceEngine(model_name_0, max_len=args.tree_max_len)
    elif draft_engine_class.lower() in ("padded", "inferenceenginepadded"):
        draft_engine = engine.InferenceEnginePadded(model_name_0, max_len=args.tree_max_len)
    elif draft_engine_class.lower() in ("er", "regular", "engineregular"):
        draft_engine = engine.EngineRegular(model_name_0, max_len=args.tree_max_len)
    else:
        raise ValueError(f"Unsupported engine class: {draft_engine_class} !")

    #logger.info(f"Loading Model 1: `{model_name_1}`")
    gptq_max_input_length = 16384  # constant for GPTQ models

    if offload:
        if "gptq" in model_name_1.lower():
            model_1 = load_gptq_offloaded_model(model_name_1, device_size=device_size, main_device=device, max_input_length=gptq_max_input_length)
        else:
            model_1 = load_offloaded_model(model_name_1, device_size=device_size, main_device=device)

    else:
        model_1 = transformers.AutoModelForCausalLM.from_pretrained(model_name_1, device_map=device, torch_dtype=torch.float16, revision=rev_1)

        if "gptq" in model_name_1.lower():
            model_1_config = transformers.AutoConfig.from_pretrained(model_name_1)
            if getattr(model_1_config.quantization_config, "act_order", False) and (model_1_config.config.max_length < 16384):
                try:
                    from auto_gptq import exllama_set_max_input_length

                    model_1 = exllama_set_max_input_length(model_1, gptq_max_input_length)
                    print("set `exllama_set_max_input_length` OK")
                except (AttributeError, ValueError, ImportError):
                    # AttributeError may happen if GPTQ-quantized model has no attribute 'device_to_buffers'
                    # could be fixed by using code from post_init()
                    # ImportError resembles https://github.com/open-mmlab/mmdetection3d/issues/1152
                    pass
                    #logger.warning("Failed to set `exllama_set_max_input_length`")

    # target_engine = EngineStatic(model_1, max_len=args.tree_max_len)
    target_engine = engine.EngineRegular(model_1, max_len=args.tree_max_len)

    if gen_type.lower() in ("sx_base", "base", "sx2", "spec_exec_base", "specexecbase"):
        spec_generator = SpecExecBase(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("spec_exec_beams", "specexecbeams", "sx_beams"):
        spec_generator = SpecExecBeams(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("sa", "a", "spec_adaptive", "specadaptive"):
        spec_generator = SpecAdaptive(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("sf", "f", "spec_fixed", "specfixed"):
        spec_generator = SpecFixed(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("si", "spec_infer", "specinfer"):
        spec_generator = SpecInfer(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("sis", "spec_infer_stems", "specinferstems"):
        spec_generator = SpecInferStems(draft_engine, target_engine, tokenizer)
    else:
        raise ValueError(f"unknown {gen_type=}")

    #logger.info(f"Created spec_generator of type {gen_type}; Models: {model_name_0}, {model_name_1}")
    return spec_generator
