#!/usr/bin/env bash
#python node2vec_classify.py --data_name mutagenesis
#python node2vec_classify.py --data_name hepatitis
#python node2vec_classify.py --data_name mondial
#python node2vec_classify.py --data_name genes
#python node2vec_classify.py --data_name world

for r in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    echo Mutagenesis ${r}
    python node2vec_dynamic.py --data_name mutagenesis --train_ratio ${r}

    echo Hepatitis ${r}
    python node2vec_dynamic.py --data_name hepatitis --train_ratio ${r}

    echo Mondial ${r}
    python node2vec_dynamic.py --data_name mondial --train_ratio ${r}

    echo Genes ${r}
    python node2vec_dynamic.py --data_name genes --train_ratio ${r}

    echo World ${r}
    python node2vec_dynamic.py --data_name world --train_ratio ${r}
done;

