from operator import index
from pydantic import BaseModel

from typing import List, Optional


class QueryInput(BaseModel):
    query: str
    model_name: str
    option: int


class InstagramData(BaseModel):
    name: Optional[str]
    country: Optional[str]
    full_name: Optional[str]
    bio: Optional[str]
    follows: Optional[int]
    following: Optional[int]
    tags: Optional[List[str]]


class SearchResult(BaseModel):
    index: int
    distance: float
    instagram_data: Optional[InstagramData]


class SearchResponse(BaseModel):
    results: List[SearchResult]
