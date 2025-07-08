def apply_for_mmlu_redux(sample: dict) -> list:
    hint = 'Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'
    question = sample['question']
    choices = sample['choices']
    choices_template = list()
    for i, c in enumerate(choices):
        choices_template.append(f"{chr(ord('A')+i)}. {c}")
    choices_template = "\n".join(choices_template)
    msg = f"{hint}\nQuestion: {question}\n{choices_template}\nanswer: "
    return [
        {
            "role": "user",
            "content": msg
        }
    ]


TEMPLATES = {
    "mmlu-redux": apply_for_mmlu_redux
}