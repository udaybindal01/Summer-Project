import argparse
import os
import pickle
import datasets
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import logging
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import tempfile
import shutil
import random
import numpy as np
from torch.cuda.amp import GradScaler
from bert_score import score as bert_score_func

class SummarizationDataset(Dataset):
    def __init__(self, tokenizer,filename):
        
        self.model_name = "allenai/led-large-16384"

        self.tokenizer = tokenizer
        self.name = filename
        
        with open(self.name, "rb") as f:
            self.data = pickle.load(f)
        
        
        self.max_input_len = 16384
        self.max_output_len = 1024

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        entry = self.data[idx]
        input_ids = self.tokenizer.encode(entry['script'], truncation=True, max_length=self.max_input_len,
                                          padding='max_length') 
        output_ids = self.tokenizer.encode(entry['summary'], truncation=True, max_length=self.max_output_len,
                                           padding='max_length') 

        if self.model_name =="allenai/led-large-16384":
            output_ids = output_ids[1:]
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        
        pad_token_id = 1
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids


def set_global_attention_mask(input_ids,tokenizer):
    
    global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
    global_attention_mask[:, 0] = 1
    
    global_attention_mask[(input_ids == tokenizer.convert_tokens_to_ids('INT'))] = 1
    global_attention_mask[(input_ids == tokenizer.convert_tokens_to_ids('EXT'))] = 1
    global_attention_mask[(input_ids == tokenizer.convert_tokens_to_ids('ĠINT'))] = 1
    global_attention_mask[(input_ids == tokenizer.convert_tokens_to_ids('ĠEXT'))] = 1

    return global_attention_mask

    
def train_model(num_training_steps,model,optimizer,config,tokenizer,scaler):
    progress_bar = tqdm(range(num_training_steps))
    completed_steps = 0
    step = 0
    best_val_r1 = -1
    print(device)
    epochs = 40
    fp16 = 'store_true'
    accum = 1
    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        for batch in train_dataloader:

            input_ids = batch[0].to(device)
            output_ids = batch[1].to(device)
            outputs = model(input_ids, attention_mask=(input_ids != tokenizer.pad_token_id),  # mask padding tokens
                            global_attention_mask=set_global_attention_mask(input_ids,tokenizer),  # set global attention
                            labels=output_ids, use_cache=False)

            loss = outputs.loss
            loss = loss / accum

            train_loss += loss.item()
            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % 10 ==0:
                # log_metrics(step, {'lr': get_lr(), 'steps': step,'epochs': epoch, "optimize_steps": completed_steps,
                #                    'loss/train': loss.item(), "running_train_loss": train_loss})

                print(f"{step}, lr: {optimizer.param_groups[0]['lr']} epoch: {epoch}, optimizer_step: {completed_steps}")
                
            if step % accum == 0:

                if fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

                completed_steps += 1

                train_loss = 0.0

            # if step % args.val_every == 0:

            #     logger.info(f'Evaluating and saving model at epoch:{epoch} step: {step}')
            #     log_metrics(step, {'lr': get_lr(), 'steps': step,'epochs': epoch, "optimize_steps": completed_steps,
            #                        'loss/train': loss.item(), "running_train_loss": train_loss})
            #     val_metrics = evaluate_step(model,args,tokenizer)
            #     val_metrics["steps"] = step
            #     log_metrics(step, val_metrics)
            #     if val_metrics["val_rouge1"] > best_val_r1:

            #         logger.info(f'Metric improved')
            #         save_checkpoint(epoch, step, model, args,config,scaler,lr_scheduler,completed_steps, best=True)
            #         best_val_r1 = val_metrics["val_rouge1"]
            #     else:
            #         save_checkpoint(epoch, step, model, args,config,scaler,lr_scheduler,completed_steps, best=False)

                model.train()

            step += 1
            progress_bar.update(1)
        # logger.info(f'Saving model checkpoint at end of epoch:{epoch} step: {step - 1}')
        # save_checkpoint(epoch, step, model, args,config,scaler,lr_scheduler,completed_steps, best=False)

    print("END OF TRAINING\n")
    
    
def evaluate_step(model,args,tokenizer, test=False):
    
    model.eval()

    if test:
        data_loader = test_dataloader
        all_predictions = []
        all_references = []
    else:
        data_loader = validation_dataloader

    metricsDict = {}
    count=0
    for batch in data_loader:
        input_ids = batch[0].to(device)
        output_ids = batch[1].to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids,
                                           attention_mask=(input_ids != tokenizer.pad_token_id),
                                           global_attention_mask=set_global_attention_mask(input_ids,tokenizer),
                                           use_cache=True, max_length=1024, num_beams=4)

        # Convert predicted and gold token ids to strings
        predictions = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        if test:
            all_predictions += predictions
            all_references += references

        # Compute rouge
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        if test:
            results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
            bertscore.add_batch(predictions=predictions, references=references)

        else:
            results = rouge.compute(predictions=predictions, references=references)

        for metric_name in metric_names:

            metric_val = input_ids.new_zeros(1) + results[metric_name].mid.fmeasure
            if metric_name in metricsDict:
                metricsDict[metric_name].append(metric_val)
            else:
                metricsDict[metric_name] = [metric_val]

        count+=1


    metricsDictReturn = {}
    for metric_name in metric_names:
        if test:
            metricsDictReturn["test_" + metric_name] = torch.mean(torch.cat(metricsDict[metric_name])).item()
        else:
            metricsDictReturn["val_" + metric_name] = torch.mean(torch.cat(metricsDict[metric_name])).item()

    if test:
        bert_result = bertscore.compute(lang="en",batch_size=4)
        metricsDictReturn["test_bert_p"] = np.mean(bert_result["precision"])
        metricsDictReturn["test_bert_r"] = np.mean(bert_result["recall"])
        metricsDictReturn["test_bert_f"] = np.mean(bert_result["f1"])

    if test:
        with open(os.path.join(args.test_summaries,"pred.pkl"), "wb") as f:
            pickle.dump(all_predictions, f)

        with open(os.path.join(args.test_summaries,"ref.pkl"), "wb") as f:
            pickle.dump(all_references, f)

    return metricsDictReturn

def add_model_specific_args(parser):
    parser.add_argument("--seed", type=int, default=1234, help="Seed")
    parser.add_argument("--lr", type=float, default=5e-5, help="Maximum learning rate")
    return parser

if __name__ == '__main__':
    
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = add_model_specific_args(main_arg_parser)
    args = parser.parse_args()
    args.test_summaries = os.path.join(args.output_dir, "test_summaries")
    
    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.test_summaries):
        os.makedirs(args.test_summaries)

    
    rouge = datasets.load_metric('rouge')
    bertscore = datasets.load_metric("bertscore")
    
    cache_dir = os.path.expanduser("~/transformers_cache")


    os.makedirs(cache_dir, exist_ok=True)


    config = AutoConfig.from_pretrained("allenai/led-large-16384", cache_dir=cache_dir)
    config.forced_bos_token_id = 0
    config.gradient_checkpointing = True 
    config.use_cache = False

    config.attention_window = [1024] * len(config.attention_window)


    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384", config=config, cache_dir=cache_dir)


    tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384", use_fast=True, cache_dir=cache_dir)
    
    model.resize_token_embeddings(len(tokenizer))
    
    print("Starting\n")
    
    trainDataset = SummarizationDataset(tokenizer, "train.pkl")
    valDataset = SummarizationDataset(tokenizer, "val.pkl")
    testDataset = SummarizationDataset(tokenizer, "test.pkl")
    
    
    train_dataloader = DataLoader(trainDataset, batch_size=4, shuffle=True,
                                  num_workers=4,
                                  collate_fn=SummarizationDataset.collate_fn)
    validation_dataloader = DataLoader(valDataset, batch_size=4, shuffle=False,
                                       num_workers=4,
                                       collate_fn=SummarizationDataset.collate_fn)
    test_dataloader = DataLoader(testDataset, batch_size=4, shuffle=False,
                                 num_workers=4,
                                 collate_fn=SummarizationDataset.collate_fn)
    
    num_training_steps = 40*len(train_dataloader)
    
    scaler = GradScaler(enabled='store_true')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1024,
                                                   num_training_steps=num_training_steps)
    
    model.to(device)
    
    train_model(num_training_steps, model, optimizer, config,tokenizer,scaler)
    
    print("TESTING\n")
    test_metric = evaluate_step(model,args,tokenizer, test=True)
    print("Metric: ",test_metric)
    
    

    
    
    