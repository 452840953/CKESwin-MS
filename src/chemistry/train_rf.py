"""Portable RF fitting/export adapter around the archived specimen-grouped helpers.

The default fits the paper-selected RF on non-test specimens. Optional search
uses the original RF search space and GroupShuffleSplit protocol. Input spectra
must already be binned at 20 mmu and thresholded at 3%; this is not a raw-spectrum
binning implementation.
"""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, help='Header CSV: label, specimen ID, metadata, then binned features.')
    parser.add_argument('--test-list', required=True, help='Held-out specimen list; one ID per line, optional tab/comma class.')
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[2] / 'configs/rf.json'))
    parser.add_argument('--output', default='outputs/ms_rf')
    parser.add_argument('--search', choices=['fixed', 'bayes', 'random'], default='fixed')
    parser.add_argument('--expected-features', type=int, default=1827)
    args = parser.parse_args()

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import LabelEncoder
    from chemistry import rf_helpers as h

    table = h.read_csv_auto_encoding(args.csv, header=0)
    if table.shape[1] < 4:
        parser.error('Expected label, specimen ID, metadata and feature columns.')
    labels = table.iloc[:, 0].astype(str)
    groups = table.iloc[:, 1].astype(str).to_numpy()
    numeric = table.iloc[:, 3:].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric.to_numpy()).all():
        parser.error('Provide finite numeric binned intensities. Missing-value imputation must be fitted on non-test data.')
    features = h.ensure_numeric_df(numeric)
    if features.shape[1] != args.expected_features:
        parser.error(f'Expected {args.expected_features} features, got {features.shape[1]}.')
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    heldout = h._read_fixed_eval_groups(args.test_list)
    if not heldout or not heldout.issubset(set(groups)):
        parser.error('The held-out list must be nonempty and every ID must occur in the input CSV.')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    h.FIXED_EVAL_LIST = args.test_list
    Xtr, ytr, gtr, Xtest, ytest, cv = h.train_eval_split_fixed_by_groups(
        features.to_numpy(), y, groups, heldout, inner_train_ratio=0.75,
        random_state=42, print_info=True, out_dir=str(out), label_decoder=list(encoder.classes_))
    if set(ytr) != set(y):
        parser.error('Every species must occur in the non-test training pool.')
    h.assert_no_leak(gtr, groups[np.isin(groups, list(heldout))], cv, Xtr, ytr)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.search == 'fixed':
        estimator = RandomForestClassifier(**config).fit(Xtr, ytr)
    else:
        rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42, n_jobs=-1)
        if args.search == 'bayes':
            from skopt import BayesSearchCV
            from skopt.space import Integer, Categorical
            search = BayesSearchCV(rf, search_spaces={
                'n_estimators': Integer(100, 1000), 'max_depth': Integer(2, 30),
                'min_samples_split': Integer(2, 20), 'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2', None])},
                n_iter=50, cv=cv, n_jobs=-1, random_state=42, verbose=0, scoring='accuracy')
        else:
            search = RandomizedSearchCV(rf, param_distributions={
                'n_estimators': np.arange(100, 1001, 50), 'max_depth': np.arange(2, 31, 2).tolist() + [None],
                'min_samples_split': np.arange(2, 21), 'min_samples_leaf': np.arange(1, 11),
                'max_features': ['sqrt', 'log2', None]},
                n_iter=60, cv=cv, n_jobs=-1, random_state=42, verbose=1, scoring='accuracy')
        search.fit(Xtr, ytr, groups=gtr)
        estimator = search.best_estimator_
    h.save_json(estimator.get_params(), out / 'rf_config_used.json')
    h.save_json(h.compute_metrics(ytest, estimator.predict(Xtest), estimator.predict_proba(Xtest)), out / 'heldout_metrics.json')
    joblib.dump(estimator, out / 'rf.pkl')
    probabilities = estimator.predict_proba(features.to_numpy())
    predictions = estimator.predict(features.to_numpy()).astype(int)
    exported = pd.DataFrame({'specimen_id': groups, 'species': labels,
                             'pred_species': encoder.inverse_transform(predictions)})
    for c in range(len(encoder.classes_)):
        exported[f'p{c}'] = probabilities[:, c]
    exported.to_csv(out / 'ms_probabilities.csv', index=False, encoding='utf-8')
    h.save_json({f'p{i}': label for i, label in enumerate(encoder.classes_)}, out / 'probability_class_order.json')
    print(f'RF and probabilities written to {out}. Verify class order against the visual graph class mapping before fusion.')

if __name__ == '__main__':
    main()
