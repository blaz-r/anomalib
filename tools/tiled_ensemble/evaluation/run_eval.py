"""Script used to go over all models and dataset categories, then train and eval performance."""

import gc
import sys
from datetime import datetime
from pathlib import Path
import csv

import traceback

import torch
from omegaconf import OmegaConf

sys.path.append(".")

from tools.tiled_ensemble.train_ensemble import get_parser as parser_ens, train as train_ens
from tools.train import get_parser as parser_one, train as train_one


MODEL_CONFIGS_PATH = Path("tools/tiled_ensemble/evaluation/configs/model")
DATASETS_PATH = Path("datasets")
MODIFIED_CONFIGS_PATH = Path("tools/tiled_ensemble/evaluation/configs/modified")
RESULTS_PATH = Path("tools/tiled_ensemble/evaluation/results")
ERRORS_PATH = Path("tools/tiled_ensemble/evaluation/errors")
ENS_CONFIG_PATH = Path("tools/tiled_ensemble/evaluation/configs/ens_config.yaml")


def train_all_from_conf():
    benchmark_config = OmegaConf.load("tools/tiled_ensemble/evaluation/configs/eval_config.yaml")

    MODIFIED_CONFIGS_PATH.mkdir(exist_ok=True)
    RESULTS_PATH.mkdir(exist_ok=True)
    ERRORS_PATH.mkdir(exist_ok=True)

    csv_path = RESULTS_PATH / "results.csv"
    csv_exists = csv_path.exists()

    # create csv file and write header
    with open(csv_path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")

        if not csv_exists:
            writer.writerow(
                [
                    "architecture",
                    "setup",
                    "dataset",
                    "category",
                    "image_F1Score",
                    "image_AUROC",
                    "pixel_F1Score",
                    "pixel_AUROC",
                    "pixel_AUPRO",
                    "total_time",
                    "test_time",
                    "cuda_memory"
                ]
            )

        for cycle, seed in enumerate(benchmark_config.seeds):
            for dataset in benchmark_config.datasets:
                for cat in benchmark_config.datasets[dataset]:
                    for arch in benchmark_config.architectures:
                        for setup, setup_settings in benchmark_config.setups.items():
                            print(f"Training {arch}[{setup}] on {dataset} - {cat}")

                            model_config = OmegaConf.load(MODEL_CONFIGS_PATH / f"{arch}.yaml")

                            # model_config.dataset.num_workers = 0
                            model_config.project.seed = seed
                            model_config.dataset.format = dataset
                            model_config.dataset.name = f"{dataset}_{setup}"
                            model_config.dataset.category = cat
                            model_config.dataset.image_size = setup_settings["image_size"]
                            model_config.dataset.path = f"./datasets/{dataset}"
                            # only save imgs for first seed run
                            model_config.visualization.save_images = (cycle == 0)

                            if setup_settings["basic_tiling"]:
                                model_config.dataset.tiling.apply = True
                                model_config.dataset.tiling.tile_size = setup_settings["tile_size"]
                                model_config.dataset.tiling.stride = setup_settings["stride"]
                            else:
                                model_config.dataset.tiling.apply = False

                            current_model_config_path = MODIFIED_CONFIGS_PATH / setup / f"{arch}.yaml"
                            OmegaConf.save(model_config, current_model_config_path)

                            if setup.startswith("ens"):
                                # tiled ensemble setup
                                ens_config = OmegaConf.load(ENS_CONFIG_PATH)
                                ens_config.tiling.tile_size = setup_settings["tile_size"]
                                ens_config.tiling.stride = setup_settings["stride"]

                                # only save imgs for first seed run
                                ens_config.visualization.save_images = (cycle == 0)

                                current_ens_config_path = MODIFIED_CONFIGS_PATH / setup / "ens_config.yaml"
                                OmegaConf.save(ens_config, current_ens_config_path)

                                arglist = ["--model_config", str(current_model_config_path),
                                           "--ensemble_config", str(current_ens_config_path)]

                                args = parser_ens().parse_args(arglist)
                                train_function = train_ens
                            else:
                                arglist = ["--config", str(current_model_config_path)]

                                args = parser_one().parse_args(arglist)
                                train_function = train_one

                            try:
                                torch._C._cuda_clearCublasWorkspaces()
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()

                                print("Cleared ->>>", torch.cuda.max_memory_reserved())

                                results = train_function(args)

                                # reserved bytes.all.peak
                                results["cuda_memory"] = torch.cuda.max_memory_reserved() / 10**6

                                torch._C._cuda_clearCublasWorkspaces()
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()

                                writer.writerow(
                                    [
                                        arch,
                                        setup,
                                        dataset,
                                        cat,
                                        results["image_F1Score"],
                                        results["image_AUROC"],
                                        results["pixel_F1Score"],
                                        results["pixel_AUROC"],
                                        results["pixel_AUPRO"],
                                        results["total_time"],
                                        results["test_time"],
                                        results["cuda_memory"],
                                    ]
                                )
                                csv_file.flush()
                            except Exception:
                                with open(ERRORS_PATH / f"{arch}_err.log", "a") as err_file:
                                    err_file.write(f"{datetime.now()} - [{setup} - {dataset} - {cat}] {traceback.format_exc()}\n")
                                    print(traceback.format_exc())


if __name__ == "__main__":
    train_all_from_conf()
