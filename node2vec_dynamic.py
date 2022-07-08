import argparse
import numpy as np
import networkx as nx
import torch.utils.data
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db_utils import Database
import os
import json

import node2vec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_node2vec_embedding_new(G, epochs=5, channels=100):
    embedding, model = node2vec.node2vec_embedding(G, 40, 30, 5, embed_dim=channels, neg_samples=20, batch_size=40000,
                                                   epochs=epochs)
    embedding = {n: embedding[i] for i, n in enumerate(G.nodes())}
    return embedding, model


def compute_embedding(db):
    G = db.get_row_val_graph()
    embedding, model = get_node2vec_embedding_new(G, epochs=5)
    return embedding, model


def dynamic_neighbors_embedding(new_db, embedding, feature_size=100):
    row_nodes = [row_id for _, row_id, _ in new_db.iter_rows()]
    for rel_id, row_id, row in new_db.iter_rows():
        count = 0
        curr_embedding = np.zeros(feature_size)
        for col, cell in row.items():
            col_id = f'{col}@{rel_id}'
            if not col_id == new_db.predict_col:
                values = cell.split() if type(cell) == str else [cell]
                for val in values:
                    val_id = f'{val}@{col_id}'
                    if val_id in embedding:
                        curr_embedding = np.add(curr_embedding, embedding[val_id])
                    else:
                        curr_embedding = np.add(curr_embedding, np.float32(np.random.normal(0.0, 1.0, feature_size)))
                    count += 1
        embedding[row_id] = np.true_divide(curr_embedding, count) if count > 0 else curr_embedding
    return embedding


def dynamic_similar_tuples_embedding(new_db, embedding, old_db, feature_size=100):
    G_old = old_db.get_row_val_graph()
    G_new = new_db.get_row_val_graph()
    row_nodes = [row_id for _, row_id, _ in new_db.iter_rows()]
    for rel_id, row_id, row in new_db.iter_rows():
        neighbors = list(G_new[row_id])
        similar_tuples = [list(G_old[neighbor]) for neighbor in neighbors if neighbor in embedding]
        similar_tuples = [inner for outer in similar_tuples for inner in outer]  # unites to one big list
        similar_tuples = list(set(similar_tuples))  # unique
        similar_tuples_filtered = []
        count = 0
        for t in similar_tuples:
            t_neighbors = list(G_old[t])
            common_per = len(list(set(neighbors) & set(t_neighbors))) / len(neighbors)
            if common_per > 0.3:
                count += 1
                similar_tuples_filtered.append(t)
        similar_tuples_embedding = np.array([np.array(embedding[x]) for x in similar_tuples_filtered])
        embedding[row_id] = np.mean(similar_tuples_embedding, axis=0) if count > 0 else np.float32(
            np.random.normal(0.0, 1.0, feature_size))

    return embedding


def dynamic_similar_tuples_weighted_embedding(new_db, embedding, old_db, feature_size=20):
    G_old = old_db.get_row_val_graph()
    G_new = new_db.get_row_val_graph()
    row_nodes = [row_id for _, row_id, _ in new_db.iter_rows()]
    for rel_id, row_id, row in new_db.iter_rows():
        neighbors = list(G_new[row_id])
        similar_tuples = [list(G_old[neighbor]) for neighbor in neighbors if neighbor in embedding]
        similar_tuples = [inner for outer in similar_tuples for inner in outer]  # unites to one big list
        similar_tuples = list(set(similar_tuples))  # unique
        similar_tuples_a = []
        count = 0
        for t in similar_tuples:
            t_neighbors = list(G_old[t])
            common_per = len(list(set(neighbors) & set(t_neighbors))) / len(neighbors)
            similar_tuples_a.append(common_per)

        similar_tuples_a /= np.sum(similar_tuples_a)
        similar_tuples_embedding = np.array(
            [np.array(embedding[x]) * a_i for x, a_i in zip(similar_tuples, similar_tuples_a)])
        embedding[row_id] = np.mean(similar_tuples_embedding, axis=0) if count > 0 else np.float32(
            np.random.normal(0.0, 1.0, feature_size))

    return embedding

def pre_process_map(partition_map, g):
    new_map = {i: partition_map[n] for i, n in enumerate(g.nodes())}
    return new_map

def dynamic_gradient_embedding(db, num_partitions):
    G, partition_map = db.get_row_val_graph(partition=num_partitions)
    partition_map = pre_process_map(partition_map, G)
    embedding, model, time = node2vec.node2vec_dynamic_embedding(G, 40, 30, 5, partition_map, embed_dim=100, neg_samples=20, batch_size=40000, num_partitions=num_partitions)
    embedding = {n: embedding[i] for i, n in enumerate(G.nodes())}
    return embedding, model, time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ration of train data")
    parser.add_argument("--data_name", type=str, default='genes', help="Name of the data base")
    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)

    name = args.data_name
    path = f'Datasets/{name}'

    db = Database.load_csv(path)
    Y, rows = db.get_labels()

    ratio = args.train_ratio

    scores = []
    split = StratifiedShuffleSplit(train_size=ratio, random_state=0, n_splits=10)
    for i, (train_index, test_index) in enumerate(split.split(rows, Y)):
        train_rows = [rows[j] for j in train_index]
        test_rows = [rows[j] for j in test_index]
        #partition = {**{t: 0 for t in train_rows}, **{t: 1 for t in test_rows}}
        partition = {**{t: 0 for t in train_rows}, **{t: 1+i for i, t in enumerate(test_rows)}}
        num_partitions = len(test_rows) + 1
        #num_partitions = 2
        db.partition(partition=partition)

        embedding, _, time = dynamic_gradient_embedding(db, num_partitions)

        X_train = np.float32([embedding[rows[j]] for j in train_index])
        X_test = np.float32([embedding[rows[j]] for j in test_index])
        Y_train, Y_test = [Y[j] for j in train_index], [Y[j] for j in test_index]

        model = SVC(random_state=1, max_iter=300)
        clf = make_pipeline(StandardScaler(), model)
        clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)
        time_per_tuple = float(time/len(test_rows))
        print(f'Ratio {ratio}, Run {i}: {score}, Seconds per Tuple: {time_per_tuple:.4f}')
        scores.append(score)

    print(f"Ratio {ratio} Result: {np.mean(scores):.4f} (+-{np.std(scores):.4f})")

    res_path = f'Results/{args.data_name}_N2V_{args.train_ratio}.json'
    os.makedirs('Results', exist_ok=True)

    with open(res_path, 'w') as f:
        json.dump({'Acc': float(np.mean(scores)), 'std': float(np.std(scores))}, f, indent=4)
