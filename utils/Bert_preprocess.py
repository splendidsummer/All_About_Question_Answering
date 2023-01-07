from datasets import load_dataset, load_metric
datasets = load_dataset("squad_v2" if squad_v2 else "squad")
