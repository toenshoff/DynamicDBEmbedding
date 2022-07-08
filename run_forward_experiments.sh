#!/usr/bin/env bash
python forward.py --classifier SVM --data_name mutagenesis --kernel EK --depth 3 --dim 100 --num_samples 5000 --epochs 10 --batch_size 50000
python forward.py --classifier SVM --data_name mutagenesis --kernel MMD --depth 3 --dim 100 --num_samples 5000 --epochs 25 --batch_size 5000
python forward.py --classifier SVM --data_name hepatitis --kernel EK --depth 3 --dim 100 --num_samples 5000 --epochs 25 --batch_size 50000
python forward.py --classifier SVM --data_name hepatitis --kernel MMD --depth 3 --dim 100 --num_samples 5000 --epochs 5 --batch_size 5000
python forward.py --classifier SVM --data_name mondial --kernel EK --depth 2 --dim 100 --num_samples 5000 --epochs 25 --batch_size 50000
python forward.py --classifier SVM --data_name mondial --kernel MMD --depth 2 --dim 100 --num_samples 5000 --epochs 25 --batch_size 5000
python forward.py --classifier SVM --data_name genes --kernel EK --depth 3 --dim 100 --num_samples 1000 --epochs 25 --batch_size 10000
python forward.py --classifier SVM --data_name genes --kernel MMD --depth 3 --dim 100 --num_samples 1000 --epochs 25 --batch_size 5000
python forward.py --classifier SVM --data_name world --kernel EK --depth 3 --dim 100 --num_samples 1000 --epochs 50 --batch_size 5000
python forward.py --classifier SVM --data_name world --kernel MMD --depth 3 --dim 100 --num_samples 5000 --epochs 50 --batch_size 5000

for r in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    echo Mutagenesis ${r}
    python forward_inductive.py --classifier SVM --data_name mutagenesis --kernel EK --depth 3 --dim 100 --num_samples 5000 --epochs 10 --batch_size 50000 --train_ratio ${r}
    python forward_inductive.py --classifier SVM --data_name mutagenesis --kernel MMD --depth 3 --dim 100 --num_samples 5000 --epochs 25 --batch_size 5000 --train_ratio ${r}

    echo Hepatitis ${r}
    python forward_inductive.py --classifier SVM --data_name hepatitis --kernel EK --depth 2 --dim 100 --num_samples 5000 --epochs 10 --batch_size 50000 --train_ratio ${r}
    python forward_inductive.py --classifier SVM --data_name hepatitis --kernel MMD --depth 2 --dim 100 --num_samples 5000 --epochs 25 --batch_size 5000 --train_ratio ${r}

    echo Mondial ${r}
    python forward_inductive.py --classifier SVM --data_name mondial --kernel EK --depth 3 --dim 100 --num_samples 5000 --epochs 10 --batch_size 50000 --train_ratio ${r}
    python forward_inductive.py --classifier SVM --data_name mondial --kernel MMD --depth 3 --dim 100 --num_samples 5000 --epochs 25 --batch_size 5000 --train_ratio ${r}

    echo Genes ${r}
    python forward_inductive.py --classifier SVM --data_name genes --kernel EK --depth 2 --dim 100 --num_samples 1000 --epochs 10 --batch_size 10000 --train_ratio ${r}
    python forward_inductive.py --classifier SVM --data_name genes --kernel MMD --depth 2 --dim 100 --num_samples 1000 --epochs 25 --batch_size 5000 --train_ratio ${r}

    echo World ${r}
    python forward_inductive.py --classifier SVM --data_name world --kernel EK --depth 3 --dim 100 --num_samples 1000 --epochs 10 --batch_size 5000 --train_ratio ${r}
    python forward_inductive.py --classifier SVM --data_name world --kernel MMD --depth 3 --dim 100 --num_samples 5000 --epochs 25 --batch_size 5000 --train_ratio ${r}
done;

