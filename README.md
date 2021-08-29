# FIL: General Field-aware Interaction Learning for Click-through Rate Prediction

## 1. How to Run

1. Download the datasets, including Criteo, Avazu and MovieLens-1M.

2. Preprocess the dataset and split it to train, valid and test set. Take criteo as an example:
```bash
python preprocess.py --dataset=criteo --raw_file=data/criteo/train.txt --target_dir=data/criteo
```

3. Transform the dataset to TFRecord dataset for training.
```bash
python data_utils.py --dataset=criteo
```

4. Change parameters according to the dataset, such as `batch_size`, `sparse_field_num`, `sparse_feature_size` and so on.
In addition, to speed up the training, please turn off the `run_eagerly` setting in `compile` function.

5. Train the model:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python main.py --do_train
```

6. Evaluate the model:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python main.py --do_eval
```

## 2. Used in Other Models

There are two ways to apply the FIL module to any model. which can be represented by the following two pieces of pseudo code. You can get the FIL module from `layers.py/FilLayer`.

1. Use the FIL module to generate field-aware embeddings.
```python
embeddings = feature_preprocess_fn(features)  # (batch_size, feature_num, feature_dim)
embeddings = FIL_Module(embeddings)  # Now, we get field-aware feature representation produced by the FIL module.
logits = any_model(embeddings)  # Use the resulted embeddings in arbitrary models.
```


2. Use the FIL module as a branch of the model to calculate logits directly.
```python
embeddings = feature_preprocess_fn(features)  # (batch_size, feature_num, feature_dim)
other_logits = any_model(embeddings)  # (batch_size, 1)
fil_logits = Dense(1)(FIL_Module(embeddings))  # （batch_size, 1)
logits = other_logits + fil_logits  # Balance the importance of the FIL module and other models.
```
In order to get better results, we recommend processing numerical features into vectors, please refer to the AutoInt model.

## 3. Requirements

* tensorflow>=2.2.0
* pandas
* scikit-learn
