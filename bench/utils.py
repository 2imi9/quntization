import json

from omegaconf import DictConfig, OmegaConf


def write_jsonl(samples: list[dict], path: str) -> None:
    with open(path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def save_config(cfg: DictConfig, path: str) -> None:
    yaml_str = OmegaConf.to_yaml(cfg)
    with open(path, "w") as f:
        f.write(yaml_str)
