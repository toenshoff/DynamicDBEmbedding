import numpy as np
import os
import torch
from tqdm import tqdm
import argparse
from io_utils import save_dict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from db_utils import Database
from forward import Forward, train, preproc_data, kernels
import mmd_utils
import ek_utlis
import json
from timeit import default_timer as timer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_samples(db, num_samples, scheme_tuple_map, tuples_left=None, tuples_right=None, col_scheme_filter=None):
    samples = {}
    for scheme, tuple_map in scheme_tuple_map.items():
        cur_rel = scheme.split(">")[-1]
        if len(db.rel_comp_cols[cur_rel]) > 0:
            for col_id in db.rel_comp_cols[cur_rel]:
                if col_scheme_filter is None or f'{scheme}>{col_id}' in col_scheme_filter:
                    col_kernel = kernels[db.get_col_type(col_id)]
                    pairs, values = sample_fct(db, col_id, tuple_map, num_samples, col_kernel, tuples_left=tuples_left, tuples_right=tuples_right)

                    if pairs is not None:
                        full_scheme = f"{scheme}>{col_id}"
                        samples[full_scheme] = (pairs, values)

    return samples


def compute_initial_embedding(db, dim, batch_size, epochs):
    tuples = [r for _, r, _ in db.iter_rows(db.predict_rel, partition=0)]
    scheme_tuple_map = db.scheme_tuple_map(db.predict_rel, tuples, args.depth, partition=0)

    samples = get_samples(db, args.num_samples, scheme_tuple_map)

    row_idx = {r: i for i, (_, r, _) in enumerate(db.iter_rows(db.predict_rel, partition=0))}
    scheme_idx = {s: i for i, s in enumerate(samples.keys())}
    model = Forward(dim, len(samples), row_idx, scheme_idx)

    loader = preproc_data(samples, model, batch_size)
    train(model, loader, epochs)

    embedding = model.get_embedding()
    embedding = {r: embedding[i] for r, i in row_idx.items()}
    return embedding, model, scheme_tuple_map


def infer(model, db, old_rows, new_rows, scheme_tuple_map):
    model.to(device)

    new_embedding = {}
    print('sampling paths for new tuples...')
    for i, t in enumerate(tqdm(new_rows)):
        t_map = db.scheme_tuple_map(db.predict_rel, [t], args.depth, partition=(1+i))
        for s, m in t_map.items():
            if s in scheme_tuple_map.keys():
                scheme_tuple_map[s][t] = t_map[s][t]
            #else:
            #    scheme_tuple_map[s] = t_map[s]

        #print("sampling for inductive embeddings...")
        samples = get_samples(db, args.num_samples_inductive, scheme_tuple_map, [t], old_rows, col_scheme_filter=set(model.scheme_idx.keys()))

        # stack pairs of tuples and map them to integer indices
        pairs = np.vstack([p for p, _ in samples.values()])
        idx_old = torch.tensor(model.row_idx(pairs[:, 1]))

        # stack kernel values
        vals = torch.tensor(np.concatenate([v for _, v in samples.values()], axis=0))

        # stack schemes and map them to integer indices
        scheme = [np.int64([model.scheme_idx[s]] * samples[s][0].shape[0]) for s in samples.keys()]
        scheme = torch.tensor(np.concatenate(scheme, axis=0))

        #print("computing new embeddings...")
        idx = np.where(pairs[:, 0] == t)[0]
        x = model.infer(idx_old[idx], scheme[idx], vals[idx])
        new_embedding[t] = x
        model.append(x, t)
        old_rows.append(t)

    return new_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='genes', help="Name of the data base")
    parser.add_argument("--dim", type=int, default=100, help="Dimension of the embedding")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the walks")
    parser.add_argument("--kernel", type=str, default='EK', choices={'EK', 'MMD'}, help="Kernel to use for ForWaRD")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples per start tuple and metapath")
    parser.add_argument("--num_samples_inductive", type=int, default=2000, help="Number of samples per start tuple and metapath used for the inductive step")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size during training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs during training")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ration of train data")
    parser.add_argument("--reps", type=int, default=5, help="Test repetitions")
    parser.add_argument("--classifier", type=str, default='SVM', choices={'NN', 'SVM'}, help="Downstream Classifier")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = f'models/dynamic/{args.data_name}/{args.kernel}_{args.depth}_{args.dim}_{args.num_samples}_{args.epochs}_{args.batch_size}/{args.train_ratio}'
    os.makedirs(model_dir, exist_ok=True)

    data_path = f'Datasets/{args.data_name}'
    db = Database.load_csv(data_path)

    sample_fct = ek_utlis.ek_sample_fct if args.kernel == 'EK' else mmd_utils.mmd_sample_fct

    Y, rows = db.get_labels()

    scores = []
    split = StratifiedShuffleSplit(train_size=args.train_ratio, random_state=0, n_splits=10)
    for i, (train_index, test_index) in enumerate(split.split(rows, Y)):

        train_rows = [rows[i] for i in train_index]
        test_rows = [rows[i] for i in test_index]
        partition = {**{t: 0 for t in train_rows}, **{t: 1+i for i, t in enumerate(test_rows)}}
        db.partition(partition=partition)

        train_embedding, model, scheme_tuple_map = compute_initial_embedding(db, args.dim, args.batch_size, args.epochs)

        X_train = np.float32([train_embedding[row] for row in train_rows])
        Y_train, Y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]

        clf = MLPClassifier(max_iter=1000) if args.classifier == 'NN' else SVC(kernel='rbf', C=1.0)
        clf = make_pipeline(StandardScaler(), clf)

        clf.fit(X_train, Y_train)

        start = timer()
        test_embedding = infer(model, db, train_rows, test_rows, scheme_tuple_map)
        end = timer()

        X_test = np.float32([test_embedding[row] for row in test_rows])
        score = clf.score(X_test, Y_test)

        scores.append(float(score))
        save_dict({'scores': scores}, f'{model_dir}/results.json')
        time_per_tuple = (end - start) / len(test_rows)

        print(f"Run {i}; Accuracy: {score:.2f}, Time Per New Tuple: {time_per_tuple:4f}")

    print(f'Acc: {np.mean(scores):.4f} (+-{np.std(scores):.4f})')

    res_path = f'Results/{args.data_name}_{args.kernel}_{args.train_ratio}.json'
    os.makedirs('Results', exist_ok=True)

    with open(res_path, 'w') as f:
        json.dump({'Acc': float(np.mean(scores)), 'std': float(np.std(scores))}, f, indent=4)
