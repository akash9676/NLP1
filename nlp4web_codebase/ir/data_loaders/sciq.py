from typing import Dict, List
from nlp4web_codebase.ir.data_loaders import IRDataset, Split
from nlp4web_codebase.ir.data_loaders.dm import Document, Query, QRel
from datasets import load_dataset
import joblib


@(joblib.Memory(".cache").cache)
def load_sciq(verbose: bool = False) -> IRDataset:
    train = load_dataset("allenai/sciq", split="train")
    validation = load_dataset("allenai/sciq", split="validation")
    test = load_dataset("allenai/sciq", split="test")
    data = {Split.train: train, Split.dev: validation, Split.test: test}

    # Each duplicated record is the same to each other:
    df = train.to_pandas() + validation.to_pandas() + test.to_pandas()
    for question, group in df.groupby("question"):
        assert len(set(group["support"].tolist())) == len(group)
        assert len(set(group["correct_answer"].tolist())) == len(group)

    # Build:
    corpus = []
    queries = []
    split2qrels: Dict[str, List[dict]] = {}
    question2id = {}
    support2id = {}
    for split, rows in data.items():
        if verbose:
            print(f"|raw_{split}|", len(rows))
        split2qrels[split] = []
        for i, row in enumerate(rows):
            example_id = f"{split}-{i}"
            support: str = row["support"]
            if len(support.strip()) == 0:
                continue
            question = row["question"]
            if len(support.strip()) == 0:
                continue
            if support in support2id:
                continue
            else:
                support2id[support] = example_id
            if question in question2id:
                continue
            else:
                question2id[question] = example_id
            doc = {"collection_id": example_id, "text": support}
            query = {"query_id": example_id, "text": row["question"]}
            qrel = {
                "query_id": example_id,
                "collection_id": example_id,
                "relevance": 1,
                "answer": row["correct_answer"],
            }
            corpus.append(Document(**doc))
            queries.append(Query(**query))
            split2qrels[split].append(QRel(**qrel))

    # Assembly and return:
    return IRDataset(corpus=corpus, queries=queries, split2qrels=split2qrels)


if __name__ == "__main__":
    # python -m nlp4web_codebase.ir.data_loaders.sciq
    import ujson
    import time

    start = time.time()
    dataset = load_sciq(verbose=True)
    print(f"Loading costs: {time.time() - start}s")
    print(ujson.dumps(dataset.get_stats(), indent=4))
    # ________________________________________________________________________________
    # [Memory] Calling __main__--home-kwang-research-nlp4web-ir-exercise-nlp4web-nlp4web-ir-data_loaders-sciq.load_sciq...
    # load_sciq(verbose=True)
    # |raw_train| 11679
    # |raw_dev| 1000
    # |raw_test| 1000
    # ________________________________________________________load_sciq - 7.3s, 0.1min
    # Loading costs: 7.260092735290527s
    # {
    #     "|corpus|": 12160,
    #     "|queries|": 12160,
    #     "|qrels-train|": 10409,
    #     "|qrels-dev|": 875,
    #     "|qrels-test|": 876
    # }
