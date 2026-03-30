"""
Standalone script to pre-build the knowledge graph.
Run this before processing the dataset:
  python build_kg.py
"""
from helpers.knowledge_graph import build_graph

RESULTS_DIRS = [
    "../../pllm_results",
    "../../pyego-results",
    "../../readpy-results",
]

if __name__ == "__main__":
    build_graph(RESULTS_DIRS, "/app/knowledge_graph.db")
    print("Knowledge graph built.")
