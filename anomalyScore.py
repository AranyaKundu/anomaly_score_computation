# import general python libraries
import os, shutil, pathlib, time
import numpy as np, pandas as pd
from matplotlib import pyplot as plt # required only for data visualization
from numpy import save, load
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import transformers as ppb
import torch

# Set class for models

class RunModel():
    def __init__(self, model_name, *args):
        self.model_name = model_name
        self.args = args

    def chooseModel(self):
        if self.model_name == 'DistilBERT':
            model, tokenizer = self.distilbertModel(**self.args)
        elif self.model_name == 'AlBERT':
            model, tokenizer = self.albertModel(**self.args)
        elif self.model_name == 'tinyBERT':
            model, tokenizer = self.tinybertModel(**self.args)
        else: raise ValueError("Invalid Model Name or Model not used!")

    # Configure the model based on user settings
    def distilbertModel(self, **kwargs):
        # All info to be entered by the user
        """
            Sets the configuration of the distilBERT Model for downstream Analysis, Returns model and tokenizer with the user specified settings
        """
        layers = kwargs.get('layers', 3)
        hidden_size = kwargs.get('hidden_size', 256)
        num_attention_heads = kwargs.get('num_attention_heads', 4)
        max_length = kwargs.get('max_length', 512)
        # Configure the distilBERT Model
        config = ppb.DistilBertConfig(num_hidden_layers = layers, hidden_size = hidden_size, 
                                        num_attention_heads = num_attention_heads, max_position_embeddings=max_length)
        model = ppb.DistilBertModel(config).from_pretrained('distilbert-base-uncased')
        tokenizer = ppb.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # Return model and tokenizer with the set parameters
        return model, tokenizer

    def albertModel(self, **kwargs):

        """
            Sets the configuration of the AlBERT Model for downstream Analysis. Returns model and tokenizer with the user specified settings
        """
        attention_probs_dropout_prob = kwargs.get('attention_probs_dropout_prob', 0.16)
        classifier_dropout_prob = kwargs.get('classifier_dropout_prob', 0.16)
        hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.16)
        hidden_size = kwargs.get('hidden_size', 256)
        num_attention_heads = kwargs.get('num_attention_heads', 4)
        num_hidden_layers = kwargs.get('num_hidden_layers', 3)
        max_length = kwargs.get('max_length', 512)
        # Set all configurations for the AlBERT model
        config = ppb.AlbertConfig.from_pretrained('albert-base-v1')
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.classifier_dropout_prob = classifier_dropout_prob
        config.hidden_dropout_prob = hidden_dropout_prob
        config.hidden_size = hidden_size
        config.num_attention_heads = num_attention_heads
        config.num_hidden_layers = num_hidden_layers
        config.max_position_embeddings = max_length
        model = ppb.AlbertModel(config).from_pretrained('albert-base-v1')
        tokenizer = ppb.AlbertTokenizer.from_pretrained('albert-base-v1')

        return model, tokenizer


    def tinybertModel(self, **kwargs):

        """
            Sets the configuration of the tinyBERT Model for downstream Analysis. Returns model and tokenizer with the user specified settings.
        """
        modelname = kwargs.get('modelname', "sentence-transformers/paraphrase-TinyBERT-L6-v2")
        max_length = kwargs.get('max_length', 512)
        tokenizer = ppb.AutoTokenizer.from_pretrained(modelname)
        tokenizer.model_max_length = max_length
        model = ppb.AutoModel.from_pretrained(modelname)

        return model, tokenizer

# Write a function for generating and saving embeddings
def create_embeddings(model, tokenizer, data, num_splits = 10, batch_size = 128, sleep_time = 10, folderpath = "", output_file = "combined.npy"):
    if data.empty:
        raise ValueError("Missing Input Data! Check your input data. This is necessary for creating embeddings.")
    
    else:
        # Generate the encodings
        encodings = tokenizer.batch_encode_plus(list(data), truncation = True, padding = True)
    
    # Create Input ids and Attention Masks
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])

    # Split the data to save RAM: decide number of splits according to your data size and RAM availability
    split_input_ids = np.array_split(input_ids, num_splits)
    split_attention_mask = np.array_split(attention_mask, num_splits)

    if folderpath is not None:
    # Generate the embeddings: Train embeddings
        for i in tqdm(range(len(split_input_ids))):
            batch_size = batch_size
            embeddings = []

            for j in range(0, len(split_input_ids[i]), batch_size):

                batch_input_ids = split_input_ids[i][j:j+batch_size]
                batch_attention_mask = split_attention_mask[i][j:j+batch_size]

                with torch.no_grad():

                    batch_outputs = model(batch_input_ids, batch_attention_mask)
                    batch_embeddings = batch_outputs[0][:, 0, :].numpy()
                    embeddings.append(batch_embeddings)
                
            embeddings = np.concatenate(embeddings, axis = 0)

            # create file to save for each chunk in the data. Number of chunks depends on the number of splits
            filename = f"embeddings_{i}.npy"
            save(os.path.join(folderpath, filename), embeddings)
            time.sleep(sleep_time)

        
        all_embeddings = []
        for name in os.listdir(folderpath):

            if "embeddings" in name and name.endswith(".npy"):

                embeddings = np.load(os.path.join(folderpath, name))
                all_embeddings.append(embeddings)
                
        combined_embeddings = np.concatenate(all_embeddings)
        np.save(os.path.join(folderpath, output_file), combined_embeddings)
    
    else: raise ValueError("Provide folderpath for saving the embeddings files.")


def anomalyScore(train_file, test_file, folderpath):
    if train_file is not None or test_file is not None or folderpath is not None:
        train_file = os.path.join(folderpath, train_file)
        test_file = os.path.join(folderpath, test_file)
        test_embeddings = np.load(test_file)
        train_embeddings = np.load(train_file)

    else: raise ValueError("At least one or more inputs are missing!")
    
    anomaly_scores = []
    for i in tqdm(range(len(test_embeddings))):
        similarities = cosine_similarity([test_embeddings[i]], train_embeddings)[0]
        anomaly_score = 1 - similarities.max()
        anomaly_scores.append(anomaly_score)
    
    return np.array(anomaly_scores)