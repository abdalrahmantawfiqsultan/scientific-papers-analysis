from pydantic import BaseModel, Field, ConfigDict
from typing import List

class Method(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(description="The exact name of the method or technique")
    category: str = Field(description="The general category of the method (e.g., Deep Learning, Statistics)")

class Researcher(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(description="The full name of the author")

class Dataset(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(description="The name of the dataset")
    
class ResearchProblem(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(description="The scientific problem or task being addressed")

class Result(BaseModel):
    model_config = ConfigDict(graph_id_fields=["description"])
    description: str = Field(description="The scientific result or finding")
    improvement: str = Field(description="The improvement over baseline, if any", default="")

class Metric(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(description="The evaluation metric used")
    value: str = Field(description="The reported value or score")

class ScientificPaper(BaseModel):
    model_config = ConfigDict(graph_id_fields=["title", "year"])
    title: str = Field(description="The exact title of the paper")
    abstract: str = Field(description="A short 2-3 sentence abstract/description")
    year: int = Field(description="The publication year")
    doi: str = Field(description="The DOI of the paper if available", default="")
    
    # Relationships
    authors: List[Researcher] = Field(description="Researchers who authored this paper", default_factory=list)
    uses_methods: List[Method] = Field(description="Methods or techniques used in this paper", default_factory=list)
    uses_datasets: List[Dataset] = Field(description="Datasets used for evaluation or training", default_factory=list)
    addresses_problems: List[ResearchProblem] = Field(description="Research problems this paper addresses", default_factory=list)
    evaluated_by: List[Metric] = Field(description="Metrics used to evaluate the results", default_factory=list)
    reports_results: List[Result] = Field(description="Key scientific results reported", default_factory=list)
    
    cites: List[str] = Field(description="Titles of papers this paper explicitly CITES", default_factory=list)
    builds_on: List[str] = Field(description="Titles of papers this paper BUILDS_ON", default_factory=list)
    extends: List[str] = Field(description="Titles of papers this paper EXTENDS", default_factory=list)
    compares_to: List[str] = Field(description="Titles of papers this paper COMPARES to", default_factory=list)
    contradicts: List[str] = Field(description="Titles of papers this paper CONTRADICTS", default_factory=list)
