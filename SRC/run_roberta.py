import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import nltk
import string
import re
import json
import pandas as pd
from transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import TensorDataset
import numpy as np
from torch import nn
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, RobertaModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import random
from tqdm import tqdm

df_news = pd.read_csv("../data/fake_news.csv")
df_news = df_news.drop(axis='columns', columns=["mid", "keywords", "titles", "news_urls", "fact_check_urls"])
df_news["label"] = 1
df_news = df_news.rename(columns={"mis": "tweet"})


df_vax_tweets = pd.read_csv("../data/vax_tweets.csv", on_bad_lines='skip')
df_vax_tweets = df_vax_tweets[["id", "text"]]



df_vax_labels = pd.read_csv("../data/fake_vax_labels.csv")
df_vax_tweets = df_vax_tweets.merge(df_vax_labels, on="id")
df_vax_tweets = df_vax_tweets.drop(axis="columns", columns="id")
df_vax_tweets = df_vax_tweets.rename(columns={"text": "tweet", "is_misinfo": "label"})

df_tweets = pd.read_csv("../data/fake_tweets.csv")
df_tweets["label"] = df_tweets["label"].apply(lambda x: 0 if x == "real" else 1)
df_tweets = df_tweets.drop(axis="columns", columns="id")


df_org_tweets = pd.DataFrame(columns=["edit_history_tweets", "id", "text"])
org = ["who", "cdc"]
cdc_pages=[4, 9]
who_pages = [9]
for i in range(2):
    for user in org:
        if user == "who" and i == 1: continue
        if user == "who":
            it = who_pages
        else:
            it = cdc_pages
        for page in range(it[i]):
            fname = f"data/{user}_page{page+1}_date{i+1}.txt"
            print(f"Loading " + fname)
            try:
                data = json.load(open(fname))
            except:
                with open(fname) as f:
                    data = f.read()
                data = json.loads(data)
            df_org_tweets = pd.concat([df_org_tweets, pd.DataFrame(data["data"])])

df_org_tweets = df_org_tweets.drop(axis="columns", columns=["id", "edit_history_tweet_ids", "edit_history_tweets"])
df_org_tweets["label"] = 0
df_org_tweets = df_org_tweets.rename(columns={"text": "tweet"})

df_comb = pd.concat([df_news, df_vax_tweets, df_tweets, df_org_tweets])
df_comb.to_csv("../data/full_fake_dataset.csv")

large, test = train_test_split(df_comb, test_size=0.1, random_state=12345)
train, val = train_test_split(large, test_size=0.1, random_state=12345)

sentiments = {"Real": 0, "Fake": 1}

def get_weights():
  class_weights = []
  for sent in sentiments:
    class_weights.append(1)

  for label in train["label"]:
    if label == 0: class_weights[0] += 1
    elif label == 1: class_weights[1] += 1
  
  print(class_weights)

  for idx, weight in enumerate(class_weights):
    class_weights[idx] = 1/weight

  dataset_weights = []
  for label in train["label"]:
    dataset_weights.append(class_weights[label])

  return dataset_weights

dataset_weights = get_weights()

nltk.download("stopwords")
nltk.download("punkt")

class TDataset(Dataset):
    def __init__(self, df, tokenizer) -> None:
        texts = df.tweet.values.tolist()
        texts = [self.preprocess_data(str(t)) for t in texts]
        self.texts = [tokenizer(t, padding="max_length", max_length = 280, truncation=True, return_tensors="pt") for t in texts]

        if "label" in df:
            self.labels = df.label.values.tolist()
    
    def preprocess_data(self, tweet):
        tweet = tweet.replace("&amp;", " ")
        tweet = re.sub(r'(@.*?)[\s]', ' ', tweet)
        tweet = re.sub(r'\s+', ' ', tweet)
        tweet = re.sub(r'^RT[\s]+', ' ', tweet)
        tweet = re.sub(r'https?:\/\/[^\s\n\r]+', ' ', tweet)
        tweet = re.sub(r'#', ' ', tweet)
        tweet = ''.join(character for character in tweet if character not in string.punctuation)
        # tweet = " ".join(text_tokens)

        tokenized_tweet = nltk.word_tokenize(tweet, language="english")
        stop_words = nltk.corpus.stopwords.words('english')
        tokenized_tweet = [t for t in tokenized_tweet if t not in stop_words]
        tweet_c = " ".join(tokenized_tweet)

        return tweet_c.strip()
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        return text, label


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


class Classifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

train_model = RobertaModel.from_pretrained("roberta-base")
model = Classifier(train_model)

batch_size = 4


def get_weights(data):
  class_weights = []
  for sent in sentiments:
    class_weights.append(1)

  for label in data:
    if label == 0: class_weights[0] += 1
    elif label == 1: class_weights[1] += 1
    else: class_weights[2] += 1
  
  print(class_weights)

  for idx, weight in enumerate(class_weights):
    class_weights[idx] = 1/weight

  dataset_weights = []
  for label in data:
    dataset_weights.append(class_weights[label])

  return dataset_weights

train_dataset_weights = get_weights(train["label"])
val_dataset_weights = get_weights(val["label"])
test_dataset_weights = get_weights(test["label"])
# print(len(train_dataset_weights), len(val_dataset_weights), len(test_dataset_weights))
dataloader_train = DataLoader(
    TDataset(train, tokenizer),
    sampler=WeightedRandomSampler(train_dataset_weights, num_samples = len(train_dataset_weights), replacement=True),
    batch_size=16
)

dataloader_val = DataLoader(
    TDataset(val, tokenizer),
#     sampler=RandomSampler(dataset_val),
    sampler=WeightedRandomSampler(val_dataset_weights, num_samples = len(val_dataset_weights), replacement=True),
    batch_size=32
)

dataloader_test = DataLoader(
    TDataset(test, tokenizer),
#     sampler = RandomSampler(dataset_test),
    sampler=WeightedRandomSampler(test_dataset_weights, num_samples = len(test_dataset_weights), replacement=True),
    batch_size = 32
)

optimizer = AdamW(
    model.parameters(),
    lr = 1e-5,
    eps = 1e-8
)

epochs = 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps = len(dataloader_train)*epochs
)


def accuracy(preds, labels):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

def b_accuracy(preds, labels):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    return balanced_accuracy_score(labels_flat, preds_flat)

def precision(preds, labels):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, preds_flat, average = 'weighted')

def recall(preds, labels):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, preds_flat, average = 'weighted')

def f1_score_func(preds, labels):
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted'), f1_score(labels_flat, preds_flat, average = 'macro')

def accuracy_per_class(preds, labels):
    sentiments_inverse = {v: k for k, v in sentiments.items()}
    
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {sentiments_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if (device == 'cuda'): torch.cuda.empty_cache()

seed_val = 12345
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# train_model.load_state_dict(torch.load("Models/Pace_models/DistilBert/BERT_ft_Epoch10_n.model"))
criterion = nn.BCELoss()

model.to(device)
criterion.to(device)
print(device)

def evaluate(dataloader_val, name=None):
    
#     if name:
#         model.load(name)
    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    confidence = []
    
    for val_input, val_label in tqdm(dataloader_val):
        
        attention_mask = val_input['attention_mask'].to(device)
        input_ids = val_input['input_ids'].squeeze(1).to(device)

        val_label = val_label.to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask)

        loss = criterion(output, val_label.float().unsqueeze(1))

        loss_val_total += loss.item()
#         print(output)
        logits = (output >= 0.5).int()
        # acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
        # total_acc_val += acc


        logits = logits.detach().cpu().numpy().reshape(-1, )
        label_ids = val_label.cpu().numpy()
        # cat_ids = inputs['category'].cpu().numpy()
#         print(logits)
        confidence.append(output.detach().cpu().numpy().reshape(-1,))
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    confidence = np.concatenate(confidence, axis=0)
            
    return loss_val_avg, predictions, true_vals, confidence

for epoch in tqdm(range(0, epochs+1)):
    model.train()
    loss_train_total = 0
    total_acc_train = 0
    progress_bar = tqdm(dataloader_train, 
                        desc='Epoch {:1d}'.format(epoch), 
                        leave=False, 
                        disable=False)
    
    for train_input, train_label in progress_bar:
#         print(train_label)
        attention_mask = train_input['attention_mask'].to(device)
        input_ids = train_input['input_ids'].squeeze(1).to(device)

        train_label = train_label.to(device)

        output = model(input_ids, attention_mask)

        loss = criterion(output, train_label.float().unsqueeze(1))

        loss_train_total += loss.item()
#         print(output)
#         print(train_label)
        acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
        total_acc_train += acc

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(train_input))})     
    
    torch.save(model.state_dict(), f'{epoch}_n.model')
#     model.save_pretrained('Models/Pace_models/DistilBert/')
    
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')
    tqdm.write(f'Something acc: {total_acc_train}')
    val_loss, predictions, true_vals = evaluate(dataloader_val)
    print(predictions.shape, true_vals.shape)
    print(predictions, true_vals)
    val_accuracy = accuracy(predictions, true_vals)
    val_b_accuracy = b_accuracy(predictions, true_vals)
    val_precision = precision(predictions, true_vals)
    val_recall = recall(predictions, true_vals)
    val_f1, val_f1_macro = f1_score_func(predictions, true_vals)
    print()
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'Accuracy: {val_accuracy}')
    tqdm.write(f'Balanced Accuracy: {val_b_accuracy}')
    tqdm.write(f'Precision: {val_precision}')
    tqdm.write(f'Recall: {val_recall}')
    tqdm.write(f'F1 Score (weighted): {val_f1}')
    tqdm.write(f'F1 Score (macro): {val_f1_macro}')
    print("------------------------")
    accuracy_per_class(predictions, true_vals)
    print("------------------------")
    print()


accuracy_per_class(predictions, true_vals)


def infer(loader):
    
#     model = torch.load(name)
    model.eval()
    
    predictions, confidence = [], []
    
    for val_input in tqdm(loader):
        for data in val_input:
            print(data)
            attention_mask = data['attention_mask'].to(device)
            input_ids = data['input_ids'].squeeze(1).to(device)

            with torch.no_grad():
                output = model(input_ids, attention_mask)

            logits = (output >= 0.5).int()

            logits = logits.detach().cpu().numpy().reshape(-1, )
            confidence.append(output.detach().cpu().numpy().reshape(-1, ))
            predictions.append(logits)
        
    predictions = np.concatenate(predictions, axis=0)
    confidence = np.concatenate(true_vals, axis=0)
            
    return predictions, confidence


df_test = pd.read_csv("test_batch_processed_draft2.csv")
df_test = df_test.rename(columns={"text":"tweet"})
df_test["label"] = 0
# tweet_test = df_test["tweet"].values.tolist()
test_batch = TDataset(df_test, tokenizer)

# test_dataset_weights = get_weights(df_test["label"])
# print(len(train_dataset_weights), len(val_dataset_weights), len(test_dataset_weights))
dataloader_test_tweets = DataLoader(
    test_batch,
    batch_size=16
)

_, predictions, _, confidence = evaluate(dataloader_test_tweets)
df_test["fake"] = predictions
df_test["confidence"] = confidence
print(df_test[["tweet"], ["fake"], ["confidence"]].head(20))
# df_test.to_csv("test_processed.csv")

print(df_test[["tweet", "fake", "confidence"]].head(20))

df_test.to_csv("../data/test_processed.csv")

df_test = pd.read_csv("../data/pull_final.csv")
df_test = df_test.rename(columns={"text":"tweet"})
df_test["label"] = 0
# tweet_test = df_test["tweet"].values.tolist()
test_batch = TDataset(df_test, tokenizer)

# test_dataset_weights = get_weights(df_test["label"])
# print(len(train_dataset_weights), len(val_dataset_weights), len(test_dataset_weights))
dataloader_test_tweets = DataLoader(
    test_batch,
    batch_size=16
)
_, predictions, _, confidence = evaluate(dataloader_test_tweets)
df_test["fake"] = predictions
df_test["confidence"] = confidence
df_test.to_csv("../data/pull3fina_processed.csv")