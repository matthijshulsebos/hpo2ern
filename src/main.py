#!/usr/bin/env python3
"""
ERDERA phenopacket analysis pipeline orchestrator.

Run any combination of steps from one command. If no step flag is given,
all steps run in order.

Steps:
  --cluster     Cluster phenopackets (Jaccard/UMAP/K-Means)
  --summarise   Cohort · cluster · ERN summary report   [implies --cluster]
  --plot        2-D UMAP scatter with OMIM reference    [implies --cluster]
  --train       Train Random Forest model predicting ERN (excludes 'Not applicable')
  --predict     Assign ERNs using a saved model bundle (requires --model)

Usage:
    python src/main.py --input data/                       # all steps
    python src/main.py --input data/ --cluster --summarise
    python src/main.py --input data/ --train
    python src/main.py --input data/ --predict                                           # uses ~/.cache/find-ern/ern_rf_model.joblib
    python src/main.py --input data/ --predict --predict-missing-only                    # same, only scores 'Not applicable' samples
    python src/main.py --input data/ --predict --model /custom/path/model.joblib         # override model location
    python src/main.py --input data/ --plot --n-clusters 20

All output is written to a shared timestamped directory (or --output).
"""

import datetime
import os
import sys
import argparse
import warnings
from pathlib import Path

os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
os.environ.setdefault("KMP_WARNINGS", "FALSE")
warnings.filterwarnings("ignore", category=UserWarning)

from pyhpo import Ontology

from phenopackets import load_all_phenopackets, validate_terms, build_ancestry_matrix
from cluster_phenopackets import (
    load_omim_diseases,
    umap_fit,
    cluster_kmeans,
    select_k_by_silhouette,
    cluster_disease_distances,
    save_results as save_cluster_csv,
)
from summarise_clusters import write_summary
from plot_cluster_statistics import plot_clusters
from plot_cohort_statistics import (
    plot_ern_distribution,
    plot_submitter_distribution,
    plot_hpo_term_counts,
    plot_top_hpo_terms,
    plot_cluster_ern_heatmap,
)
from predict_ern_rf import (
    NOT_APPLICABLE,
    DEFAULT_MODEL_PATH,
    filter_by_class_size,
    build_feature_matrix,
    train_and_evaluate,
    save_results as save_train_results,
    save_model_bundle,
    predict_and_save,
)
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ERDERA phenopacket pipeline: cluster · summarise · plot · train · predict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input", "-i", required=True, type=Path, metavar="DIR",
                   help="Directory with phenopacket JSON files")
    p.add_argument("--output", "-o", type=Path, default=None, metavar="DIR",
                   help="Output directory (default: output/<timestamp>/)")

    steps = p.add_argument_group(
        "Step selection",
        "Which steps to run. If none are given, all steps run.",
    )
    steps.add_argument("--cluster", action="store_true",
                       help="Cluster phenopackets")
    steps.add_argument("--summarise", action="store_true",
                       help="Generate summary report  [implies --cluster]")
    steps.add_argument("--plot", action="store_true",
                       help="Save 2-D UMAP scatter plot  [implies --cluster]")
    steps.add_argument("--train", action="store_true",
                       help="Train ERN prediction model (samples with ERN='Not applicable' are excluded)")
    steps.add_argument("--predict", action="store_true",
                       help="Assign ERNs using a saved model bundle (uses default cache path; override with --model)")

    cl = p.add_argument_group("Clustering options")
    cl.add_argument("--n-clusters", "-n", type=int, default=10, metavar="N",
                    help="Fixed cluster count (set to 0 to auto-select via silhouette)")
    cl.add_argument("--k-min", type=int, default=2, metavar="N")
    cl.add_argument("--k-max", type=int, default=30, metavar="N")
    cl.add_argument("--umap-components", type=int, default=25, metavar="N")
    cl.add_argument("--n-neighbors", type=int, default=15, metavar="N",
                    help="UMAP neighbourhood size; lower = finer local structure")
    cl.add_argument("--min-dist", type=float, default=0.1, metavar="F")
    cl.add_argument("--top-terms", type=int, default=15, metavar="N")
    cl.add_argument("--top-omim", type=int, default=10, metavar="N")
    cl.add_argument("--save-matrix", action="store_true",
                    help="Write N×N Jaccard distance matrix to CSV")
    cl.add_argument("--save-embeddings", action="store_true",
                    help="Write UMAP cluster embedding to CSV")
    cl.add_argument("--plot-top-omim", type=int, default=1, metavar="N",
                    help="OMIM disease labels per cluster in UMAP plot")

    tr = p.add_argument_group("Training options")
    tr.add_argument("--n-estimators", type=int, default=300, metavar="N")
    tr.add_argument("--test-size", type=float, default=0.20, metavar="F")
    tr.add_argument("--cv-folds", type=int, default=5, metavar="N")
    tr.add_argument("--min-class-size", type=int, default=10, metavar="N",
                    help="Drop ERN classes smaller than N for training")
    tr.add_argument("--top-features", type=int, default=30, metavar="N")
    tr.add_argument("--no-submitter", action="store_true",
                    help="Use HPO terms only; exclude submitter features")
    tr.add_argument("--no-save-model", action="store_true",
                    help="Do not persist fitted model bundle with joblib")

    pr = p.add_argument_group("Prediction options")
    pr.add_argument("--model", type=Path, default=None, metavar="FILE",
                    help=f"Model bundle (.joblib) for --predict (default: {DEFAULT_MODEL_PATH})")
    pr.add_argument("--predict-missing-only", action="store_true",
                    help="When predicting, only score samples whose ERN is 'Not applicable'")

    return p.parse_args()


def _section(title: str) -> None:
    print(f"\n {title}")


def main() -> None:
    args = parse_args()

    # Which steps to run
    any_explicit = args.cluster or args.summarise or args.plot or args.train or args.predict
    run_all = not any_explicit
    run_cluster = run_all or args.cluster or args.summarise or args.plot
    run_summarise = run_all or args.summarise
    run_plot = run_all or args.plot
    run_train = run_all or args.train
    run_predict = args.predict  # never included in run_all; always explicit

    # Output directory
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path("output") / timestamp
    else:
        run_dir = args.output
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    plots_dir = run_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    active = [s for s, on in [
        ("cluster", run_cluster), ("summarise", run_summarise),
        ("plot", run_plot), ("train", run_train), ("predict", run_predict),
    ] if on]
    print(f"Output : {run_dir}/")
    print(f"Steps  : {' · '.join(active)}")

    _section("Loading")
    Ontology()
    samples = load_all_phenopackets(args.input)
    samples = validate_terms(samples)

    if run_cluster:
        n_input = len(samples)
        if args.n_clusters and len(samples) < args.n_clusters:
            sys.exit(
                f"ERROR: only {len(samples)} usable phenopackets but "
                f"{args.n_clusters} clusters requested."
            )

        _section("Clustering")
        X_bin, vocabulary = build_ancestry_matrix(samples)
        disease_ids, disease_names, X_disease = load_omim_diseases(vocabulary)

        umap_dim = min(args.umap_components, len(samples) - 1)
        X_umap, _ = umap_fit(X_bin, umap_dim, args.n_neighbors, args.min_dist)

        if not args.n_clusters:
            k_max = min(args.k_max, len(samples) - 1)
            n_clusters, silhouette_scores = select_k_by_silhouette(X_umap, args.k_min, k_max)
        else:
            n_clusters, silhouette_scores = args.n_clusters, None

        labels = cluster_kmeans(X_umap, n_clusters)
        dist_matrix = cluster_disease_distances(X_bin, labels, X_disease)

        output_path = data_dir / "clusters.csv"
        save_cluster_csv(samples, labels, X_bin, X_umap, output_path,
                         args.save_matrix, args.save_embeddings)

        if run_summarise:
            write_summary(
                samples, labels, X_bin, X_disease, dist_matrix,
                disease_ids, disease_names, output_path,
                n_input, silhouette_scores, args.top_terms, args.top_omim,
            )

        if run_plot:
            _section("Plotting")
            plot_ern_distribution(samples, plots_dir / "cohort_ern_distribution.png")
            plot_submitter_distribution(samples, plots_dir / "cohort_submitter_distribution.png")
            plot_hpo_term_counts(samples, plots_dir / "cohort_hpo_term_counts.png")

            term_names: dict[str, str] = {}
            for tid in {t for s in samples for t in s.terms}:
                try:
                    term_names[tid] = Ontology.get_hpo_object(tid).name
                except RuntimeError:
                    pass
            plot_top_hpo_terms(samples, plots_dir / "cohort_top_hpo_terms.png",
                               term_names=term_names)
            plot_cluster_ern_heatmap(samples, labels, plots_dir / "cohort_cluster_ern_heatmap.png")

            X_2d, reducer_2d = umap_fit(X_bin.toarray(), 2, args.n_neighbors, args.min_dist)
            X_disease_2d = reducer_2d.transform(X_disease.toarray())
            plot_clusters(
                X_2d, labels, plots_dir / "clusters.csv",
                X_disease_2d=X_disease_2d,
                dist_matrix=dist_matrix,
                disease_names=disease_names,
                top_omim_per_cluster=args.plot_top_omim,
            )

    if run_train:
        _section("Training")

        labeled = [s for s in samples if s.ern != NOT_APPLICABLE]
        n_unlabeled = len(samples) - len(labeled)
        if n_unlabeled:
            print(f"  Excluded {n_unlabeled} samples with ERN='{NOT_APPLICABLE}' (no label)")

        train_samples, class_names = filter_by_class_size(labeled, args.min_class_size)
        if len(class_names) < 2:
            print("[skip] Fewer than 2 ERN classes remain; skipping.", file=sys.stderr)
        else:
            X_hpo, vocabulary = build_ancestry_matrix(train_samples)
            X, feature_names, ohe = build_feature_matrix(
                train_samples, vocabulary, X_hpo,
                include_submitter=not args.no_submitter,
            )
            print(f"  Features : {X.shape[1]}")

            le = LabelEncoder()
            le.fit(class_names)
            y = le.transform([s.ern for s in train_samples])

            model, results = train_and_evaluate(
                X, y, class_names,
                test_size=args.test_size,
                cv_folds=args.cv_folds,
                n_estimators=args.n_estimators,
            )

            report = results["report"]
            print(f"  Test accuracy : {report['accuracy']:.4f}")
            print(f"  Test F1 macro : {report['macro avg']['f1-score']:.4f}")

            save_train_results(results, model, feature_names, class_names,
                               data_dir, plots_dir, args.top_features)

            if not args.no_save_model:
                bundle_path = save_model_bundle(
                    model, vocabulary, le, ohe,
                    include_submitter=not args.no_submitter,
                )
                if bundle_path:
                    print(f"  Model bundle  : {bundle_path}")

    if run_predict:
        _section("Predicting")
        model_path = args.model if args.model is not None else DEFAULT_MODEL_PATH
        print(f"  Model bundle  : {model_path}")
        if not model_path.exists():
            print(
                f"[skip] Model bundle not found: {model_path}\n"
                "       Run --train first, or specify a path with --model.",
                file=sys.stderr,
            )
        else:
            predict_and_save(
                bundle_path=model_path,
                samples=samples,
                missing_only=args.predict_missing_only,
                output_path=data_dir / "ern_predictions.csv",
            )


if __name__ == "__main__":
    main()
