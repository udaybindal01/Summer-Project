import pandas as pd
from datasets import load_dataset
import torch
from transformers import RobertaTokenizer,RobertaModel
from tqdm import tqdm
import pickle

dataset = load_dataset("rohitsaxena/MENSA")
train = dataset["train"]
# train = train[:100]
print(len(train))
validation = dataset["validation"]
test = dataset["test"]

model = RobertaModel.from_pretrained("roberta-large")
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def scene_to_embeddings(data,model,tokenizer):
    
    model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    embeddings = []
    
    print(len(data))
    print("\n")
    for i in tqdm(range(len(data))):
        
        with torch.no_grad():
            encoded_input = tokenizer(data[i]["scenes"], return_tensors='pt', padding=True, truncation=True)
            output = model(**encoded_input.to(device))
            emneddings = output.last_hidden_state[:, 0, :]
            emneddings = emneddings.detach().cpu()
        
        data[i]["labels"] = torch.tensor(data[i]["labels"])
        data[i]["embeddings"] = embeddings
        embeddings.append(data[i])
    
    return embeddings

def savePickle(filename, obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
        
train_embeddings = scene_to_embeddings(train,model,tokenizer)
val_embeddings = scene_to_embeddings(validation,model,tokenizer)
test_embeddings = scene_to_embeddings(test,model,tokenizer)



        
savePickle("train.pkl", train_embeddings)
savePickle("val.pkl", val_embeddings)
savePickle("test.pkl", test_embeddings)

             
