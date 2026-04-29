from typing import Dict, List, Tuple
import os
import time
import pprint
import csv
import cv2
import numpy as np
import torch

from bluestar.utils.save_utils import save_predict
from bluestar.utils.config_utils import prepare_config
from bluestar.utils.wandb_utils import set_wandb
from bluestar.utils.random_utils import set_seed
from bluestar.utils.dist_utils import set_dist, is_distributed_set, is_master, barrier, get_world_size
from bluestar.utils.print_utils import time_log
from bluestar.utils.param_utils import count_params
from vgd.wrapper import VGD

def inference_epoch(
        model: VGD,
        cfg: Dict,
) -> List:

    model.eval()
    torch.set_grad_enabled(False)
    result = []
    model.reset()
    # -------------------------------- data -------------------------------- #

    # -------------------------------- loss -------------------------------- #
    results = model.inference(num_images_per_prompt=cfg["model"]["sampling"]["n_samples"],
                             num_inference_steps=cfg["model"]["sampling"]["steps"],
                             guidance_scale=cfg["model"]["sampling"]["scale"],
                             style_transfer=cfg["style_transfer"],
                             object=cfg["object"])

    s = f"... samples: {int(cfg['batch_size']) * get_world_size()} " \
        f"(valid done: {100:.2f} %)"
    if is_master():
        print(s)

    if not(os.path.exists(cfg["save_dir"])):
        os.mkdir(cfg["save_dir"])

    for output in results:
        file_exists = os.path.isfile(os.path.join(cfg["save_dir"], cfg["save_file"]))

        with open(os.path.join(cfg["save_dir"], cfg["save_file"]), "a" if file_exists else "w", newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                if cfg["style_transfer"]:
                    writer.writerow(['target_image', 'initial_condition', 'obejct', 'prompt', 'seed', 'similarity'])
                elif cfg.get("data_dir"):
                    writer.writerow(['target_image', 'initial_condition', 'prompt', 'seed', 'similarity'])
                else:
                    writer.writerow(['target_prompts', 'prompt', 'seed', 'similarity'])

            if cfg["style_transfer"]:
                writer.writerow([cfg["data_dir"][0], output['initial_condition'], output['object'], output["prompt"], output["seed"], output["similarity"]])
            elif cfg.get("data_dir"):
                writer.writerow([cfg["data_dir"][0], output['initial_condition'], output["prompt"], output["seed"], output["similarity"]])
            else:
                writer.writerow([cfg["target_prompts"], output["prompt"], output["seed"], output["similarity"]])

        similarity = output["similarity"]
        print(f"...similarity: {similarity}\n")

        if cfg["data_dir"] is not None:
            org_filename = cfg["data_dir"][0].split('/')[-1]
        else:
            org_filename = output["prompt"][0]


        if not os.path.isdir(os.path.join(cfg["save_dir"], org_filename, str(output["seed"]))):
            os.makedirs(os.path.join(cfg["save_dir"], org_filename, str(output["seed"])))

        path = os.path.join(cfg["save_dir"], org_filename, str(output["seed"]))

        if not cfg["model"]["gen_prompt_only"]:
            for cnt1, image in enumerate(output["image"]):
                if cfg["style_transfer"]:
                    filename = f'seed_{output["seed"]}_{cfg["object"]}' + '_' + str(cnt1)
                else:
                    filename = f'seed_{output["seed"]}' + '_' + str(cnt1)


                cv2.imwrite(os.path.join(path, filename + '.png'), cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))


        if cfg["style_transfer"]:
            filename = f'seed_{output["seed"]}_{cfg["object"]}'
            save_predict(
                {"generated_prompt": output["prompt"], "initial_condition": output["initial_condition"],
                 "object": output["object"], "llm_input": output["input_prompt"], "similarity": output["similarity"]},
                path + f"/{filename}.json"
            )
            result += {"generated_prompt": output["prompt"], "initial_condition": output["initial_condition"],
                       "object": output["object"], "llm_input": output["input_prompt"], "similarity": output["similarity"]}

        elif cfg.get("data_dir"):
            filename = f'seed_{output["seed"]}'
            save_predict(
                {"generated_prompt": output["prompt"], "initial_condition": output["initial_condition"],
                 "llm_input": output["input_prompt"], "similarity": output["similarity"]},
                path + f"/{filename}.json"
            )
            result += {"generated_prompt": output["prompt"], "initial_condition": output["initial_condition"],
                       "llm_input": output["input_prompt"], "similarity": output["similarity"]}

        else:
            filename = f'seed_{output["seed"]}'
            save_predict(
                {"generated_prompt": output["prompt"], "target_prompts": cfg["target_prompts"],
                 "llm_input": output["input_prompt"], "similarity": output["similarity"]},
                path + f"/{filename}.json"
            )
            result += {"generated_prompt": output["prompt"], "target_prompts": cfg["target_prompts"],
                 "llm_input": output["input_prompt"], "similarity": output["similarity"]}

    return results


def run(cfg: Dict, debug: bool = False) -> None:
    # ======================================================================================== #
    # Initialize
    # ======================================================================================== #
    device, local_rank = set_dist(device_type="cuda")

    if is_master():
        pprint.pprint(cfg)  # print config to check if all arguments are correctly given.

    _ = set_wandb(cfg, force_mode="disabled" if debug else None)
    set_seed(seed=cfg["seed"] + local_rank)

    # ======================================================================================== #
    # Model
    # ======================================================================================== #
    model = VGD(cfg)
    model = model.to(device)
    if is_distributed_set():
        # model = DistributedDataParallel(model, device_ids=[local_rank], output_device=device)
        model_m = model #.module  # actual model without wrapping
    else:
        model_m = model

    if is_master():
        print(model)
        p1, p2 = count_params(model_m.parameters())
        print(f"Model parameters: {p1} tensors, {p2} elements.")

    # ======================================================================================== #
    # Evaluation
    # ======================================================================================== #

    s = time_log()
    s += f"Start validation"
    if is_master():
        print(s)

    inference_start_time = time.time()  # second
    result = inference_epoch(model_m, cfg)
    inference_time = time.time() - inference_start_time

    s = time_log()
    s += f"End validation, time: {inference_time:.3f} s\n"


    if is_master():
        print(s)

    barrier()


if __name__ == '__main__':
    args, config = prepare_config()
    run(config, args.debug)
