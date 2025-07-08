from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, \
    Dataset

from .base import BaseDataset


class MmluReduxDataset(BaseDataset):

    DATASET_NAME = "edinburgh-dawg/mmlu-redux"

    CONFIGS = [
        'anatomy', 'business_ethics', 'clinical_knowledge', 'college_chemistry', 'college_computer_science',
        'college_mathematics', 'college_medicine', 'college_physics', 'econometrics', 'electrical_engineering',
        'formal_logic', 'global_facts', 'high_school_chemistry', 'high_school_mathematics', 'high_school_physics',
        'high_school_statistics', 'human_aging', 'logical_fallacies', 'machine_learning', 'miscellaneous', 'philosophy',
        'professional_accounting', 'public_relations', 'virology', 'conceptual_physics', 'high_school_us_history',
        'astronomy', 'high_school_geography', 'high_school_macroeconomics', 'professional_law'
    ]

    @classmethod
    def load(cls, debug: bool = False) -> Dataset:
        configs = cls.CONFIGS[:1] if debug else cls.CONFIGS
        _datasets = [load_dataset(cls.DATASET_NAME, conf, split="test") for conf in tqdm(configs)]
        dataset = concatenate_datasets(_datasets)
        return dataset

    @classmethod
    def measure(cls):
        pass
