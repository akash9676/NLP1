import os
from typing import Dict, List, Optional, Protocol
import pandas as pd
import tqdm
import ujson
from nlp4web_codebase.ir.data_loaders import IRDataset


def round_dict(obj: Dict[str, float], ndigits: int = 4) -> Dict[str, float]:
    return {k: round(v, ndigits=ndigits) for k, v in obj.items()}


def sort_dict(obj: Dict[str, float], reverse: bool = True) -> Dict[str, float]:
    return dict(sorted(obj.items(), key=lambda pair: pair[1], reverse=reverse))


def save_ranking_results(
    output_dir: str,
    query_ids: List[str],
    rankings: List[Dict[str, float]],
    query_performances_lists: List[Dict[str, float]],
    cid2tweights_lists: Optional[List[Dict[str, Dict[str, float]]]] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ranking_results.jsonl")
    rows = []
    for i, (query_id, ranking, query_performances) in enumerate(
        zip(query_ids, rankings, query_performances_lists)
    ):
        row = {
            "query_id": query_id,
            "ranking": round_dict(ranking),
            "query_performances": round_dict(query_performances),
            "cid2tweights": {},
        }
        if cid2tweights_lists is not None:
            row["cid2tweights"] = {
                cid: round_dict(tws) for cid, tws in cid2tweights_lists[i].items()
            }
        rows.append(row)
    pd.DataFrame(rows).to_json(
        output_path,
        orient="records",
        lines=True,
    )


class TermWeightingFunction(Protocol):
    def __call__(self, query: str, cid: str) -> Dict[str, float]: ...


def compare(
    dataset: IRDataset,
    results_path1: str,
    results_path2: str,
    output_dir: str,
    main_metric: str = "recip_rank",
    system1: Optional[str] = None,
    system2: Optional[str] = None,
    term_weighting_fn1: Optional[TermWeightingFunction] = None,
    term_weighting_fn2: Optional[TermWeightingFunction] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df1 = pd.read_json(results_path1, orient="records", lines=True)
    df2 = pd.read_json(results_path2, orient="records", lines=True)
    assert len(df1) == len(df2)
    all_qrels = {}
    for split in dataset.split2qrels:
        all_qrels.update(dataset.get_qrels_dict(split))
    qid2query = {query.query_id: query for query in dataset.queries}
    cid2doc = {doc.collection_id: doc for doc in dataset.corpus}
    diff_col = f"{main_metric}:qp1-qp2"
    merged = pd.merge(df1, df2, on="query_id", how="outer")
    rows = []
    for _, example in tqdm.tqdm(merged.iterrows(), desc="Comparing", total=len(merged)):
        docs = {cid: cid2doc[cid].text for cid in dict(example["ranking_x"])}
        docs.update({cid: cid2doc[cid].text for cid in dict(example["ranking_y"])})
        query_id = example["query_id"]
        row = {
            "query_id": query_id,
            "query": qid2query[query_id].text,
            diff_col: example["query_performances_x"][main_metric]
            - example["query_performances_y"][main_metric],
            "ranking1": ujson.dumps(example["ranking_x"], indent=4),
            "ranking2": ujson.dumps(example["ranking_y"], indent=4),
            "docs": ujson.dumps(docs, indent=4),
            "query_performances1": ujson.dumps(
                example["query_performances_x"], indent=4
            ),
            "query_performances2": ujson.dumps(
                example["query_performances_y"], indent=4
            ),
            "qrels": ujson.dumps(all_qrels[query_id], indent=4),
        }
        if term_weighting_fn1 is not None and term_weighting_fn2 is not None:
            all_cids = set(example["ranking_x"]) | set(example["ranking_y"])
            cid2tweights1 = {}
            cid2tweights2 = {}
            ranking1 = {}
            ranking2 = {}
            for cid in all_cids:
                tweights1 = term_weighting_fn1(query=qid2query[query_id].text, cid=cid)
                tweights2 = term_weighting_fn2(query=qid2query[query_id].text, cid=cid)
                ranking1[cid] = sum(tweights1.values())
                ranking2[cid] = sum(tweights2.values())
                cid2tweights1[cid] = tweights1
                cid2tweights2[cid] = tweights2
            ranking1 = sort_dict(ranking1)
            ranking2 = sort_dict(ranking2)
            row["ranking1"] = ujson.dumps(ranking1, indent=4)
            row["ranking2"] = ujson.dumps(ranking2, indent=4)
            cid2tweights1 = {cid: cid2tweights1[cid] for cid in ranking1}
            cid2tweights2 = {cid: cid2tweights2[cid] for cid in ranking2}
            row["cid2tweights1"] = ujson.dumps(cid2tweights1, indent=4)
            row["cid2tweights2"] = ujson.dumps(cid2tweights2, indent=4)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(by=diff_col, ascending=False)
    output_path = os.path.join(output_dir, f"compare-{system1}_vs_{system2}.tsv")
    table.to_csv(output_path, sep="\t", index=False)


# if __name__ == "__main__":
#     # python -m lecture2.bm25.analysis
#     from nlp4web_codebase.ir.data_loaders.sciq import load_sciq
#     from lecture2.bm25.bm25_retriever import BM25Retriever
#     from lecture2.bm25.tfidf_retriever import TFIDFRetriever
#     import numpy as np

#     sciq = load_sciq()
#     system1 = "bm25"
#     system2 = "tfidf"
#     results_path1 = f"output/sciq-{system1}/results/ranking_results.jsonl"
#     results_path2 = f"output/sciq-{system2}/results/ranking_results.jsonl"
#     index_dir1 = f"output/sciq-{system1}"
#     index_dir2 = f"output/sciq-{system2}"
#     compare(
#         dataset=sciq,
#         results_path1=results_path1,
#         results_path2=results_path2,
#         output_dir=f"output/sciq-{system1}_vs_{system2}",
#         system1=system1,
#         system2=system2,
#         term_weighting_fn1=BM25Retriever(index_dir1).get_term_weights,
#         term_weighting_fn2=TFIDFRetriever(index_dir2).get_term_weights,
#     )

#     # bias on #shared_terms of TFIDF:
#     df1 = pd.read_json(results_path1, orient="records", lines=True)
#     df2 = pd.read_json(results_path2, orient="records", lines=True)
#     merged = pd.merge(df1, df2, on="query_id", how="outer")
#     nterms1 = []
#     nterms2 = []
#     for _, row in merged.iterrows():
#         nterms1.append(len(list(dict(row["cid2tweights_x"]).values())[0]))
#         nterms2.append(len(list(dict(row["cid2tweights_y"]).values())[0]))
#     percentiles = (5, 25, 50, 75, 95)
#     print(system1, np.percentile(nterms1, percentiles), np.mean(nterms1).round(2))
#     print(system2, np.percentile(nterms2, percentiles), np.mean(nterms2).round(2))
#     # bm25 [ 3.  4.  5.  7. 11.] 5.64
#     # tfidf [1. 2. 3. 5. 9.] 3.58
