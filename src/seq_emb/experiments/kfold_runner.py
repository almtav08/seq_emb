"""Cross-validation experiment runner supporting Leave-One-Out (LOOCV) and Stratified K-Fold."""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from ..core.seed import set_seed, get_device
from ..data.loader import DataLoader
from ..data.processor import prepare_tensors
from ..models.factory import create_model
from ..evaluation.evaluator import evaluate_fold


class CrossValidationRunner:
    """Executes LOOCV or Stratified K-Fold evaluation across sequential models with/without POG."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        seed: int = 42,
        device: Optional[str] = "auto",
    ):
        self.seed = seed
        self.device = get_device(device)
        self.loader = DataLoader(data_dir=data_dir)
        self.data = self.loader.load_all()
        # Pre-trained RotatE weights are strictly frozen with requires_grad=False
        self.graph_embedder = self.loader.load_graph_embedder(self.device)

    def run(
        self,
        model_name: str,
        use_pog: bool,
        strategy: str = "loocv",
        n_splits: int = 5,
        num_epochs: int = 500,
        batch_size: int = 4,
        max_seq_len: int = 225,
        custom_params: Optional[Dict[str, Any]] = None,
        local_window: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run model evaluation using either Leave-One-Out Cross-Validation ('loocv')
        or Stratified K-Fold ('kfold').
        """
        strat = strategy.lower().strip()
        if strat in ["loocv", "leave_one_out", "loo"]:
            return self.run_loocv(
                model_name=model_name,
                use_pog=use_pog,
                num_epochs=num_epochs,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                custom_params=custom_params,
                local_window=local_window,
                verbose=verbose,
            )
        elif strat in ["kfold", "stratified_kfold"]:
            return self.run_kfold(
                model_name=model_name,
                use_pog=use_pog,
                n_splits=n_splits,
                num_epochs=num_epochs,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                custom_params=custom_params,
                local_window=local_window,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown validation strategy: '{strategy}'. Choose 'loocv' or 'kfold'.")

    def run_loocv(
        self,
        model_name: str,
        use_pog: bool,
        num_epochs: int = 500,
        batch_size: int = 4,
        max_seq_len: int = 225,
        custom_params: Optional[Dict[str, Any]] = None,
        local_window: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute Leave-One-Out Cross-Validation (LOOCV).
        Each student is held out as test exactly once, training on the remaining N-1 students.
        """
        set_seed(self.seed)

        paths = self.data["paths"]
        grades = self.data["grades"]
        prereq_graph = self.data["prereq_graph"]
        remedial_graph = self.data["remedial_graph"]
        popularity_map = self.data["popularity_map"]
        total_resources = self.data["total_resources"]
        num_items = self.data["num_items"]
        embedding_dim = self.graph_embedder.embedding_dim

        student_ids = list(paths.keys())
        n_students = len(student_ids)

        all_pass_metrics = {"map": [], "ndcg": [], "hr": [], "mrr": []}
        all_fail_metrics = {
            "coverage_ratio": [],
            "backward_ratio": [],
            "repeat_ratio": [],
            "progress_ratio": [],
            "frac_allowed_edges": [],
            "novelty": [],
        }

        iterator = range(n_students)
        if verbose:
            mode_str = "con POG" if use_pog else "sin POG"
            iterator = tqdm(iterator, total=n_students, desc=f"LOOCV {model_name.upper()} ({mode_str})")

        for i in iterator:
            test_key = student_ids[i]
            test_keys = [test_key]
            train_keys = [student_ids[j] for j in range(n_students) if j != i]

            # Prepare training tensors (N-1 students)
            train_x, train_y, train_masks = prepare_tensors(
                student_ids=train_keys,
                paths=paths,
                grades=grades,
                max_seq_len=max_seq_len,
                device=self.device,
                use_pog=use_pog,
                graph_embedder=self.graph_embedder if use_pog else None,
            )

            # Build model instance
            model = create_model(
                model_name=model_name,
                use_pog=use_pog,
                embedding_dim=embedding_dim,
                max_seq_length=max_seq_len,
                num_items=num_items,
                device=self.device,
                custom_params=custom_params,
            )

            # Train on N-1 students
            model.fit(
                train_x=train_x,
                train_y=train_y,
                train_masks=train_masks,
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=False,
            )

            # Evaluate on held-out student
            res = evaluate_fold(
                model=model,
                train_keys=train_keys,
                test_keys=test_keys,
                paths=paths,
                grades=grades,
                prereq_graph=prereq_graph,
                remedial_graph=remedial_graph,
                popularity_map=popularity_map,
                total_resources=total_resources,
                max_seq_len=max_seq_len,
                device=self.device,
                use_pog=use_pog,
                graph_embedder=self.graph_embedder if use_pog else None,
                local_window=local_window,
            )

            if grades[test_key] == 1:
                for k in all_pass_metrics:
                    all_pass_metrics[k].extend(res["pass_metrics"][k])
            else:
                for k in all_fail_metrics:
                    all_fail_metrics[k].extend(res["fail_metrics"][k])

        # Compute averages and standard deviations
        mean_metrics = {
            "mean_map": float(np.mean(all_pass_metrics["map"])) if all_pass_metrics["map"] else 0.0,
            "mean_ndcg": float(np.mean(all_pass_metrics["ndcg"])) if all_pass_metrics["ndcg"] else 0.0,
            "mean_hr": float(np.mean(all_pass_metrics["hr"])) if all_pass_metrics["hr"] else 0.0,
            "mean_mrr": float(np.mean(all_pass_metrics["mrr"])) if all_pass_metrics["mrr"] else 0.0,
            "mean_coverage_ratio": float(np.mean(all_fail_metrics["coverage_ratio"])) if all_fail_metrics["coverage_ratio"] else 0.0,
            "mean_backward_ratio": float(np.mean(all_fail_metrics["backward_ratio"])) if all_fail_metrics["backward_ratio"] else 0.0,
            "mean_repeat_ratio": float(np.mean(all_fail_metrics["repeat_ratio"])) if all_fail_metrics["repeat_ratio"] else 0.0,
            "mean_progress_ratio": float(np.mean(all_fail_metrics["progress_ratio"])) if all_fail_metrics["progress_ratio"] else 0.0,
            "mean_frac_allowed_edges": float(np.mean(all_fail_metrics["frac_allowed_edges"])) if all_fail_metrics["frac_allowed_edges"] else 0.0,
            "mean_novelty": float(np.mean(all_fail_metrics["novelty"])) if all_fail_metrics["novelty"] else 0.0,
        }

        std_metrics = {
            "mean_map": float(np.std(all_pass_metrics["map"], ddof=1)) if len(all_pass_metrics["map"]) > 1 else 0.0,
            "mean_ndcg": float(np.std(all_pass_metrics["ndcg"], ddof=1)) if len(all_pass_metrics["ndcg"]) > 1 else 0.0,
            "mean_hr": float(np.std(all_pass_metrics["hr"], ddof=1)) if len(all_pass_metrics["hr"]) > 1 else 0.0,
            "mean_mrr": float(np.std(all_pass_metrics["mrr"], ddof=1)) if len(all_pass_metrics["mrr"]) > 1 else 0.0,
            "mean_coverage_ratio": float(np.std(all_fail_metrics["coverage_ratio"], ddof=1)) if len(all_fail_metrics["coverage_ratio"]) > 1 else 0.0,
            "mean_backward_ratio": float(np.std(all_fail_metrics["backward_ratio"], ddof=1)) if len(all_fail_metrics["backward_ratio"]) > 1 else 0.0,
            "mean_repeat_ratio": float(np.std(all_fail_metrics["repeat_ratio"], ddof=1)) if len(all_fail_metrics["repeat_ratio"]) > 1 else 0.0,
            "mean_progress_ratio": float(np.std(all_fail_metrics["progress_ratio"], ddof=1)) if len(all_fail_metrics["progress_ratio"]) > 1 else 0.0,
            "mean_frac_allowed_edges": float(np.std(all_fail_metrics["frac_allowed_edges"], ddof=1)) if len(all_fail_metrics["frac_allowed_edges"]) > 1 else 0.0,
            "mean_novelty": float(np.std(all_fail_metrics["novelty"], ddof=1)) if len(all_fail_metrics["novelty"]) > 1 else 0.0,
        }

        return {
            "model_name": model_name,
            "use_pog": use_pog,
            "strategy": "loocv",
            "seed": self.seed,
            "n_students": n_students,
            "n_pass": len(all_pass_metrics["map"]),
            "n_fail": len(all_fail_metrics["coverage_ratio"]),
            "mean": mean_metrics,
            "std": std_metrics,
        }

    def run_kfold(
        self,
        model_name: str,
        use_pog: bool,
        n_splits: int = 5,
        num_epochs: int = 500,
        batch_size: int = 4,
        max_seq_len: int = 225,
        custom_params: Optional[Dict[str, Any]] = None,
        local_window: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run Stratified K-Fold cross-validation."""
        set_seed(self.seed)

        paths = self.data["paths"]
        grades = self.data["grades"]
        prereq_graph = self.data["prereq_graph"]
        remedial_graph = self.data["remedial_graph"]
        popularity_map = self.data["popularity_map"]
        total_resources = self.data["total_resources"]
        num_items = self.data["num_items"]
        embedding_dim = self.graph_embedder.embedding_dim

        student_ids = list(paths.keys())
        student_labels = [grades[uid] for uid in student_ids]

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        fold_results = []

        for fold, (train_loc, test_loc) in enumerate(skf.split(student_ids, student_labels)):
            if verbose:
                print(f"--- Fold [{fold + 1}/{n_splits}] ---")

            train_keys = [student_ids[i] for i in train_loc]
            test_keys = [student_ids[i] for i in test_loc]

            train_x, train_y, train_masks = prepare_tensors(
                student_ids=train_keys,
                paths=paths,
                grades=grades,
                max_seq_len=max_seq_len,
                device=self.device,
                use_pog=use_pog,
                graph_embedder=self.graph_embedder if use_pog else None,
            )

            model = create_model(
                model_name=model_name,
                use_pog=use_pog,
                embedding_dim=embedding_dim,
                max_seq_length=max_seq_len,
                num_items=num_items,
                device=self.device,
                custom_params=custom_params,
            )

            model.fit(
                train_x=train_x,
                train_y=train_y,
                train_masks=train_masks,
                num_epochs=num_epochs,
                batch_size=batch_size,
                verbose=False,
            )

            metrics = evaluate_fold(
                model=model,
                train_keys=train_keys,
                test_keys=test_keys,
                paths=paths,
                grades=grades,
                prereq_graph=prereq_graph,
                remedial_graph=remedial_graph,
                popularity_map=popularity_map,
                total_resources=total_resources,
                max_seq_len=max_seq_len,
                device=self.device,
                use_pog=use_pog,
                graph_embedder=self.graph_embedder if use_pog else None,
                local_window=local_window,
            )
            fold_results.append(metrics)

        mean_metrics = {}
        std_metrics = {}
        metric_keys = [
            "mean_map", "mean_ndcg", "mean_hr", "mean_mrr",
            "mean_coverage_ratio", "mean_backward_ratio", "mean_repeat_ratio",
            "mean_progress_ratio", "mean_frac_allowed_edges", "mean_novelty"
        ]
        for key in metric_keys:
            vals = [f[key] for f in fold_results]
            mean_metrics[key] = float(np.mean(vals))
            std_metrics[key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        return {
            "model_name": model_name,
            "use_pog": use_pog,
            "strategy": "kfold",
            "seed": self.seed,
            "n_splits": n_splits,
            "folds": fold_results,
            "mean": mean_metrics,
            "std": std_metrics,
        }


# Maintain backward compatibility alias
KFoldRunner = CrossValidationRunner
