from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    collection_id: str
    text: str


@dataclass
class Query:
    query_id: str
    text: str


@dataclass
class QRel:
    query_id: str
    collection_id: str
    relevance: int
    answer: Optional[str] = None
