# Anomaly Score Computation
## Documentation for anomalyScore.py

We can compute an Anomaly Score within a text data based on cosine similarity. It works by computing the cosine distance between two non-zero vectors. Cosine similarity is often used to measure document similarity in text analysis. The mathematical formula to compute the cosine similarity is:

$$ CosineSimilarity = \frac {(A.B)} {(||A||.||B||)} $$

<br>
where A and B are vectors:

It is better than using Euclidean distance.

For a better understanding of the same, please refer to <br>
https://medium.com/@sasi24/cosine-similarity-vs-euclidean-distance-e5d9a9375fc8#:~:text=As%20can%20be%20seen%20from,other%20than%20OA%20to%20OC

https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/

* The program is intended to compute anomaly scores between text datasets by creating classification tokens.
* After creating the classification token the data for the test dataset is compared with the whole of the train dataset to compute the cosine similarity.
Then it takes a difference of cosine similarity from 1 to get the dissimilarity or anomaly. (This is because the cosine of any value is always less than or equal to 1).
* Finally it takes the maximum of all the differences for each transcript within the dataset and stores it in a numpy array which is referred to as <b>`Anomaly Score`</b>.

## Step-by-step working process of the module

1. It imports all the necessary modules for successfully running the functions within the file.
2. There is a class (`RunModel`) followed by two different functions (`create_embeddings` \& `anomalyScore`). The class is to choose which BERT model to apply on the data set. The class is also to set some of the major parameters of the BERT model.

### How to define which model to use: Creating and instance of the RunModel Class

```python 
import anomalyScore as ASc

dbA = ASc.RunModel('distilBERT')
model, tokenizer = dbA.distilbertModel()
```

> If any parameter is to be set manually by the user:

```python
import anomalyScore as ASc

abM = ASc.RunModel('AlBERT')
model, tokenizer = abM.albertModel(max_length = 1000)
```
3. Once the model is chosen the functions can be called to create embeddings and compute the anomaly score.

4. Embeddings can be created using the create_embeddings function. We need to specify all the parameters required by the function. The docstring of the function specifies what each of the input parameters mean and why they are used.

### How to call the `create_embeddings` function? 

```python
import anomalyScore as ASc

create_embeddings(model, tokenizer, data, num_splits, batch_size, sleep_time,
folderpath, output_file)
```

> * `model` and `tokenizer` should come from defining the class. 
* `data` should be a pandas Series object or a numpy array or a python list.
* `num_splits` is to control the RAM availability. This would ensure that the function does not crash in the middle due to lack of RAM availability. The default is set to 10. However, for smaller data sets, this can be set to 1.
* `batch_time` is also a parameter to control the RAM availability. The default value is 128. For smaller data sets, the batch size can be set to the length of the data/ split, so that the whole data goes into the model all at once.
* sleep_time is a cool down time specified to ensure a difference between each split. The default time is 10 seconds, but this can be set to 0 by the user (not recommended).
* `folderpath` must be specified to run the program. This is the directory where the `embeddings` will be stored.
* `output_file` is the file in which the embeddings will be stored. The default value is `combined.npy`. Please note it is a numpy file and not a regular csv/xlsx file.

### How does the `create_embeddings` function work?

The tokenizer takes the data and generates the encodings which then becomes an input to the model. The encodings has two major components that goes to the model. These are `input_ids` and `attention_mask`.

<b>Why only `input_ids` and `attention_mask`?</b>

The distilBERT model takes various parameters such as `input_ids`, `attention_mask`, `head_mask`, `input_embeds`, `output_attentions`, `labels`. These can be changed based on the data and the method used. Since the function creates a tokenzied sequence of the data, `input_ids` are used and hence `input_embeds` are not used.

For better understanding of what they mean and how they are generated by the distilBERT tokenizer please refer to: <br>
https://huggingface.co/transformers/v3.0.2/glossary.html#input-ids

+ Attention Mask: An attention mask is a binary mask that indicates which tokens are actual words and which tokens are padding tokens. It helps the model to focus only on the relevant tokens and ignore the padding tokens during computation.

For better understanding of what they mean and how they are generated by the distilBERT tokenizer please refer to: <br>
https://huggingface.co/transformers/v3.0.2/glossary.html#attention-mask

What to extract from the generated embeddings?

```python
batch_embeddings = batch_outputs[0][:, 0, :].numpy()
```

The first token (at the 0th position) is the classification token. This is our token of consideration for any kind of analysis. There are some other kind of tokens also produced by the model. These generally includes `Special Tokens`, `Word Tokens`, `Positional Tokens`, and `Attention Mask`. 

**_Position:_** The CLS token is always placed at the beginning of the input sequence, typically as the first token. In some cases, it may be preceded by a special token, such as the SEP token in BERT, which indicates the start of the sequence.

**_Semantic Meaning:_** The CLS token is assigned a special semantic meaning by the model. During pre-training, the model is trained to understand that the CLS token represents the aggregated representation of the entire input sequence. It captures the contextual information and semantic understanding derived from the entire sequence.


### How to call the anomalyScore function?

> * The function requires three mandatory parameters (`train_data`, `test_data` & `folderpath` of the files), no optional parameters. The program will work only if the train and test data files are stored in the same folder.
* Both the train and test data files should be embeddings stored in .npy format. In other cases, it might produce unwanted results.


```python
import anomalyScore as ASc

folderpath = 'path/to/directory'
train_file = "train_data_file_name.npy"
test_file = "test_data_file_name.npy"

anomalyScore = ASc.anomalyScore(train_file, test_file, folderpath)
```

The function compares the test embeddings with the whole corpus of train embeddings and compute the cosine dissimilarity in each case. `Anomaly score` can be defined in two ways:

+ Using maximum similarity (similarities.max()): This approach considers the maximum similarity score between the test embedding and the embeddings in the training set. Subtracting this maximum similarity score from 1 gives you the `anomaly score`. In this case, a higher anomaly score indicates a higher dissimilarity or deviation from the training set.

+ Using minimum similarity (similarities.min()): This approach considers the minimum similarity score between the test embedding and the embeddings in the training set. Subtracting this minimum similarity score from 1 gives you the `anomaly score`. In this case, a lower `anomaly score` indicates a higher dissimilarity or deviation from the training set.

In this module, the anomalyScore function uses maximum simialrity only. Thus a higher `anomaly score` indicates a higher dissimilarity from the training data.

The data obtained from the function needs can be used for downstream computations.

### Precautions while running the module

+ It assumes the default value for a number of arguments of each BERT module. If you need to change any of them, read the documentation of the respective BERT module and make changes accordingly.
+ The functions are mostly designed considering the distilBERT module. It is better to read the documentation of other BERT modules to understand better about the position of the **_`CLS`_** token. The user might need to modify the code for extracting the embeddings based on the BERT used to get the `CLS` token.
