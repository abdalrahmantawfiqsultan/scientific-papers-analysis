import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
from typing import List
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

class ExtractedGraphTriplet(BaseModel):
    subject: str = Field(description="The entity doing the action (usually the paper name)")
    predicate: str = Field(description="Relationship type: USES_METHOD, STUDIES, BELONGS_TO")
    object: str = Field(description="The target entity name, e.g., 'Graph Neural Networks'")
    object_type: str = Field(description="The type of the object: Concept, Method, Field, or Dataset")
    section: str = Field(description="The section this was extracted from: Introduction, Methodology, or Related Work")

class DocumentExtractionSchema(BaseModel):
    triplets: List[ExtractedGraphTriplet]

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=4096,
    huggingfacehub_api_token=hf_token
))

chunk = "We propose a novel Graph Neural Network (GNN) for predicting Protein Folding. This approach belongs to Structural Biology."
paper_title = "Dummy Title"

prompt = f"""
Extract scientific relationships from this text chunk into structured triplets. 
The paper is titled '{paper_title}'.

Output your response strictly as a JSON object matching this schema, with no markdown formatting:
{{
    "triplets": [
        {{
            "subject": "string (entity doing the action)",
            "predicate": "string (USES_METHOD, STUDIES, BELONGS_TO)",
            "object": "string (target entity)",
            "object_type": "string (Concept, Method, Field, or Dataset)",
            "section": "string (Introduction, Methodology, or Related Work)"
        }}
    ]
}}

{chunk}
"""

print("Invoking LLM...")
raw_response = llm.invoke(prompt).content
print("Raw response:")
print(raw_response)

import json
raw_clean = raw_response.strip()
if raw_clean.startswith("```json"):
    raw_clean = raw_clean[7:]
elif raw_clean.startswith("```"):
    raw_clean = raw_clean[3:]
if raw_clean.endswith("```"):
    raw_clean = raw_clean[:-3]

print("Cleaned response:")
print(raw_clean)

try:
    parsed_json = json.loads(raw_clean.strip())
    extraction = DocumentExtractionSchema(**parsed_json)
    print("Success:", extraction)
except Exception as e:
    print("Failed:", e)
