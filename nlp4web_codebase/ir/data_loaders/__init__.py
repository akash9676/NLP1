from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from nlp4web_codebase.ir.data_loaders.dm import Document, Query, QRel


class Split(str, Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class IRDataset:
    corpus: List[Document]
    queries: List[Query]
    split2qrels: Dict[Split, List[QRel]]

    def get_stats(self) -> Dict[str, int]:
        stats = {"|corpus|": len(self.corpus), "|queries|": len(self.queries)}
        for split, qrels in self.split2qrels.items():
            stats[f"|qrels-{split}|"] = len(qrels)
        return stats

    def get_qrels_dict(self, split: Split) -> Dict[str, Dict[str, int]]:
        qrels_dict = {}
        for qrel in self.split2qrels[split]:
            qrels_dict.setdefault(qrel.query_id, {})
            qrels_dict[qrel.query_id][qrel.collection_id] = qrel.relevance
        return qrels_dict

    def get_split_queries(self, split: Split) -> List[Query]:
        qrels = self.split2qrels[split]
        qids = {qrel.query_id for qrel in qrels}
        return list(filter(lambda query: query.query_id in qids, self.queries))
