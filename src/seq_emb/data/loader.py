"""Data loading utilities for paths, grades, graphs, and pre-trained embeddings."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch


class DataLoader:
    """Loads raw data JSONs and pre-trained RotatE graph embedder."""

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Default to seq_emb/data directory
            self.base_dir = Path(__file__).resolve().parents[3] / "data"
        else:
            self.base_dir = Path(data_dir)

        self.raw_dir = self.base_dir / "raw"
        self.states_dir = self.base_dir / "states"

    def load_all(self) -> Dict[str, Any]:
        """Load all data artifacts with normalized types."""
        paths = self.load_paths()
        grades = self.load_grades()
        prereq_graph = self.load_prereq_graph()
        remedial_graph = self.load_remedial_graph()
        resources = self.load_resources()
        popularity_map = self.compute_popularity_map(paths)

        return {
            "paths": paths,
            "grades": grades,
            "prereq_graph": prereq_graph,
            "remedial_graph": remedial_graph,
            "resources": resources,
            "popularity_map": popularity_map,
            "total_resources": len(resources),
            "num_items": max(max(p) for p in paths.values()) + 1,
        }

    def load_paths(self) -> Dict[int, List[int]]:
        """Load interaction paths mapped as {student_id (int): [resource_id (int)]}."""
        filepath = self.raw_dir / "paths.json"
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): [int(item) for item in v] for k, v in raw.items()}

    def load_grades(self) -> Dict[int, int]:
        """Load student grades mapped as {student_id (int): 1 if Pass else 0}."""
        filepath = self.raw_dir / "grades.json"
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): (1 if v == "Pass" else 0) for k, v in raw.items()}

    def load_prereq_graph(self) -> Dict[int, List[int]]:
        """Load prerequisite DAG: {resource_id: [dependent_resources]}."""
        filepath = self.raw_dir / "prereq_graph.json"
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): [int(v) for v in vals] for k, vals in raw.items()}

    def load_remedial_graph(self) -> Dict[int, List[int]]:
        """Load remedial graph: {resource_id: [remedial_resources]}."""
        filepath = self.raw_dir / "remedial_graph.json"
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): [int(v) for v in vals] for k, vals in raw.items()}

    def load_resources(self) -> List[Dict[str, Any]]:
        """Load curriculum resources catalog."""
        filepath = self.raw_dir / "resources.json"
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_graph_embedder(self, device: torch.device):
        """Load pre-trained RotatE knowledge graph embedder and freeze all its parameters."""
        filepath = self.states_dir / "know.pth"
        embedder = torch.load(filepath, map_location=device, weights_only=False)
        embedder.eval()
        if hasattr(embedder, "requires_grad_"):
            embedder.requires_grad_(False)
        if hasattr(embedder, "parameters"):
            for p in embedder.parameters():
                p.requires_grad = False
        if hasattr(embedder, "to"):
            embedder.to(device)
        return embedder

    @staticmethod
    def compute_popularity_map(paths: Dict[int, List[int]]) -> Dict[int, float]:
        """Compute normalized item popularity across all paths."""
        pop = {}
        for path in paths.values():
            for item in path:
                pop[item] = pop.get(item, 0) + 1
        max_pop = max(pop.values()) if pop else 1.0
        return {k: v / max_pop for k, v in pop.items()}
