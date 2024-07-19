import torch
import pickle
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import random
import numpy as np
from torch import nn, Tensor
import math
from torch.optim import AdamW
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class SaliencyDataset(Dataset):
    
    def __init__(self,filename):
        
        self.name = filename
        with open(self.name, "rb") as f:
            self.data = pickle.load(f)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

def collate_batch(batch):
    
    max = -1
    for data in batch:
        if max<data["embeddings"].shape[0]:
            max = data["embeddings"].shape[0]
    
    scenes = torch.zeros(len(batch), max, batch[0]["embeddings"].shape[-1])
    mask = torch.ones(len(batch), max, dtype=torch.bool)
    labels = torch.ones(len(batch), max) * -1

    for i , data in enumerate(batch):
        
        embed = data["embeddings"]
        scenes[i,:len(embed),:] = embed
        mask[i,:len(embed)] = torch.zeros(len(embed), dtype=torch.bool)
        labels[i, len(embed)] = data["labels"]
    
    return scenes.to(device), mask.to(device), labels.to(device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0)]

        return x
    

class Saliency(nn.Module):
    
    def __init__(self, input_dim, nhead, num_layers, output_dim, dropout = 0.1):
        super().__init__()
        
        self.position_encoder = PositionalEncoding(input_dim,dropout,batch_first=True)
        self.encoded_layer = TransformerEncoderLayer(d_model=input_dim, nhead=nhead,  batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoded_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, embeddings, embed_mask):
        
        embeddings_pos = self.position_encoder(embeddings)
        output = self.transformer_encoder(embeddings_pos, mask = None, src_key_padding_mask = embed_mask)
        output = self.linear(output)
        return output

def getPositiveWeight(trainDataset):
    
    labels = []
    for d in range(len(trainDataset)):
        labels += trainDataset[d]["labels"]

    ones = sum(labels)
    zeros = len(labels) - ones
    positive_weight = torch.FloatTensor([zeros / ones]).to(device)
    return positive_weight

def compute_metrics(pred, target):
    
    pred = pred.long().detach().cpu()
    target = target.long().detach().cpu()

    Precision, Recall, f1, _ = precision_recall_fscore_support(target, pred, average='macro')

    acc = balanced_accuracy_score(target, pred)

    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': Precision,
        'Recall': Recall
    }
    
def averageBatch(metrics):
    acc = []
    f1=[]
    pre=[]
    recall=[]
    for idx in metrics:
        acc.append(idx['Accuracy'])
        f1.append(idx['F1'])
        pre.append(idx['Precision'])
        recall.append(idx['Recall'])

    mean_acc = np.mean(acc)
    mean_f1 = np.mean(f1)
    mean_pre = np.mean(pre)
    mean_recall = np.mean(recall)
    
    return {"Average Accuracy":mean_acc,"Average f1":mean_f1,"Average precision":mean_pre,"Average recall":mean_recall
            }

def train_model (model,criterion, optimizer, trainloader,valloader, epochs):
    
    bar = tqdm(range(epochs))
    
    for epoch in range(epochs):
        
        model.train()
        total_loss = 0
        batch_accuracy = []
        
        for scenes,mask,labels in trainloader:
            
            scenes = scenes.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            
            output = model(scenes,mask)
            loss_mask = ~mask
            loss_padded = criterion(output.squeeze(-1), labels)

            loss_unpadded = torch.masked_select(loss_padded, loss_mask)
            loss = loss_unpadded.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            
            final_output = torch.masked_select(output.squeeze(-1), loss_mask)
            pred = torch.sigmoid(final_output) > 0.5
            target = torch.masked_select(labels, loss_mask)
            batch_accuracy.append(compute_metrics(pred, target))
        
        train_loss = total_loss/ len(train_loader)
        print(f"train_loss: {train_loss}, epochs: {epoch}")
        
        model.eval()
        total_loss = 0
        batch_acc = []
        with torch.no_grad():
            for scenes, mask, labels_padded in valloader:
                scenes = scenes.to(device)
                mask = mask.to(device)
                labels_padded = labels_padded.to(device)

                output_padded = model(scenes, mask)

                loss_mask = ~mask
                loss_padded = criterion(output_padded.squeeze(-1), labels_padded)

                loss_unpadded = torch.masked_select(loss_padded, loss_mask)
                loss = loss_unpadded.mean()

                total_loss += loss.detach().item()

                pred = torch.masked_select(output_padded.squeeze(-1), loss_mask)
                pred = torch.sigmoid(pred) > 0.5

                target = torch.masked_select(labels_padded, loss_mask)
                batch_acc.append(compute_metrics(pred, target))
        
        val_loss = total_loss/len(valloader)
        val_metrics = batch_acc
        val_avg_metrics = averageBatch(val_metrics)
        print(f"val_loss: {val_loss}, epochs: {epoch}")
        print(epoch, val_avg_metrics)
        
        bar.update(1)
    
    print("End of Training\n")
       
def test_loop(model, optimizer, criterion, dataloader):
    model.eval()
    total_loss = 0
    batch_acc = []
    with torch.no_grad():
        for scenes, mask, labels_padded in dataloader:
            scenes = scenes.to(device)
            mask = mask.to(device)
            labels_padded = labels_padded.to(device)

            output_padded = model(scenes, mask)

            loss_mask = ~mask
            loss_padded = criterion(output_padded.squeeze(-1), labels_padded)

            loss_unpadded = torch.masked_select(loss_padded, loss_mask)
            loss = loss_unpadded.mean()

            total_loss += loss.detach().item()

            pred = torch.masked_select(output_padded.squeeze(-1), loss_mask)
            pred = torch.sigmoid(pred) > 0.5

            target = torch.masked_select(labels_padded, loss_mask)
            batch_acc.append(compute_metrics(pred, target))

    return total_loss / len(dataloader), batch_acc 

def generateData(model, dataloader):
    model.eval()
    dataset = []

    with torch.no_grad():
        for scenes, mask, labels_padded in dataloader:
            movieDict = {}

            scenes = scenes.to(device)
            mask = mask.to(device)
            labels_padded = labels_padded.to(device)

            output_padded = model(scenes, mask)
            loss_mask = ~mask
            pred = torch.masked_select(output_padded.squeeze(-1), loss_mask)
            pred = pred > 0.5

            movieDict["prediction_labels"] = pred.int()
            dataset.append(movieDict)
    return dataset     

def prepareDataForSummarization(pred_data, script_data):
    dataForSummarization = []
    for i in range(len(script_data)):
        movieDict = {}
        scriptTextDict = script_data[i]["scenes"]
        pred_labels = pred_data[i]["prediction_labels"].detach().cpu().tolist()

        movieDict["script"] = " ".join(scriptTextDict[idx] for idx, label in enumerate(pred_labels) if label == 1)

        movieDict["summary"] = script_data[i]["summary"]
        dataForSummarization.append(movieDict)
    return dataForSummarization

def savePickle(filename, obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)
        
def compute_similarity_graph(embeddings):
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def compute_centrality(similarity_matrix, lambda1=0.5, lambda2=0.5):
    num_scenes = similarity_matrix.shape[0]
    centrality_scores = np.zeros(num_scenes)
    
    for i in range(num_scenes):
        forward_sum = np.sum(similarity_matrix[i, (i+1):])
        backward_sum = np.sum(similarity_matrix[i, :i])
        centrality_scores[i] = lambda1 * forward_sum + lambda2 * backward_sum
    
    return centrality_scores

def select_top_k_scenes(centrality_scores, k):
    top_k_indices = np.argsort(-centrality_scores)[:k]
    return top_k_indices

def generate_predictions(dataset, k):
    predictions = []
    
    for data in dataset:
        embeddings = data["embeddings"]
        similarity_matrix = compute_similarity_graph(embeddings)
        centrality_scores = compute_centrality(similarity_matrix)
        top_k_indices = select_top_k_scenes(centrality_scores, k)
        
        prediction = np.zeros(embeddings.shape[0], dtype=int)
        prediction[top_k_indices] = 1
        predictions.append({"prediction_labels": torch.tensor(prediction)})
    
    return predictions

def combine_predictions(trans_predictions, graph_predictions):
    combined_predictions = []
    
    for trans_pred, graph_pred in zip(trans_predictions, graph_predictions):
        combined = torch.max(trans_pred["prediction_labels"], graph_pred["prediction_labels"])
        combined_predictions.append({"prediction_labels": combined})
    
    return combined_predictions

def combine_predictions(trans_predictions, graph_predictions):
    combined_predictions = []
    
    for trans_pred, graph_pred in zip(trans_predictions, graph_predictions):
        combined = torch.max(trans_pred["prediction_labels"], graph_pred["prediction_labels"])
        combined_predictions.append({"prediction_labels": combined})
    
    return combined_predictions


if __name__=='__main__':
    
    trainDataset = SaliencyDataset("train.pkl")
    valDataset = SaliencyDataset("val.pkl")
    testDataset = SaliencyDataset("test.pkl")
    
    train_loader = DataLoader(trainDataset, batch_size= 8, shuffle=True, collate_fn= collate_batch)
    valLoader = DataLoader(valDataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
    testLoader = DataLoader(testDataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

    model = Saliency(input_dim=1024, nhead = 16, num_layers=10 , output_dim=1)
    optimizer = AdamW(model.parameters(), lr = 8e-5)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=getPositiveWeight(trainDataset))
    model.to(device)
    
    EPOCHS = 25
    K = 25
    
    print("TESTING\n")
    
    test_loss, test_metrics = test_loop(model, optimizer,criterion, testLoader)
    test_metrics = averageBatch(test_metrics)
    print("FINAL: ",test_metrics)
    
    genTrainLoader = DataLoader(trainDataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    genValLoader = DataLoader(valDataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    genTestLoader = DataLoader(testDataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    
    print("Data for summarization\n")
    
    trainPred = generateData(model, genTrainLoader)
    valPred = generateData(model, genValLoader)
    testPred = generateData(model, genTestLoader)
    
    trainGraphPred = generate_predictions(trainDataset, K)
    valGraphPred = generate_predictions(valDataset, K)
    testGraphPred = generate_predictions(testDataset, K)
    
    trainCombinedPred = combine_predictions(trainPred, trainGraphPred)
    valCombinedPred = combine_predictions(valPred, valGraphPred)
    testCombinedPred = combine_predictions(testPred, testGraphPred)
    
    trainSummData = prepareDataForSummarization(trainCombinedPred,trainDataset)
    valSummData = prepareDataForSummarization(valCombinedPred,valDataset)
    testSummData = prepareDataForSummarization(testCombinedPred,testDataset)
    
    savePickle("train_data.pkl", trainSummData)
    savePickle("val_data.pkl", valSummData)
    savePickle("test_data.pkl", testSummData)
    
    print("COMPLETED\n")

