import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)



import random
import os
import re
import shutil

import n2c2_eval_script

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import codecs
# from transformers import AdamW
from torch.optim import Adam, AdamW
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch import nn


random.seed(70)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def loadData(df):
    data=[]
    tag_values = ['Disposition', 'NoDisposition', 'Undetermined']
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    for i in range(len(df)):
        data.append({'index': i,
                     'sentence': df['sentence'][i], #df['sentence'][i]
                     'label':tag2idx[df['label'][i]],
                     'drug':df['drugname'][i],
                     'span_start':df['start'][i],
                     'span_end':df['end'][i],
                     'anno_type':df['ann_type'][i],
                     'filename':df['filename'][i].split('/')[-1]
                     })
    return data




def preprocessingForBert(data,tokenizer,max_length=500):
    dataset_encoding_list=[]
    for item in data:
        # encoding=tokenizer.encode_plus("[CLS]"+item['sentence']+"[SEP]"+item['drug']+"[SEP]",
        #                                max_length=max_length,padding='max_length',truncation=True)
        encoding = tokenizer.encode_plus( item['sentence'], item['drug'],
                                          max_length=max_length, padding='max_length',
                                          truncation=True, add_special_tokens=True)
        encoding['labels']=item['label']
        encoding={k:torch.tensor(v) for k,v in encoding.items()}
        dataset_encoding_list.append(encoding)

    input_ids=torch.empty((len(dataset_encoding_list), max_length), dtype=torch.long)
    # token_type_ids=torch.empty((len(dataset_encoding_list), max_length), dtype=torch.long)
    attention_masks=torch.empty((len(dataset_encoding_list), max_length), dtype=torch.long)
    labels=torch.empty((len(dataset_encoding_list), 1), dtype=torch.float)


    for i in range(len(dataset_encoding_list)):
        input_ids[i]=torch.tensor(dataset_encoding_list[i]['input_ids'],dtype=torch.long)
        # token_type_ids[i]=torch.tensor(dataset_encoding_list[i]['token_type_ids'],dtype=torch.long)
        attention_masks[i]=torch.tensor(dataset_encoding_list[i]['attention_mask'],dtype=torch.long)
        labels[i]=torch.tensor(dataset_encoding_list[i]['labels'],dtype=torch.float)

    train_data=TensorDataset(input_ids,  attention_masks,labels) #token_type_ids,
    return train_data, len(input_ids)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed=58
g = torch.Generator()
g.manual_seed(seed)

def training(train_df,dev_df,type,epochs,model_name,tokenizer,learning_rate=0.00003,warmup_steps=0,batch_size=20):
    #  # type is to save results for different window sizes separately
    train_data=loadData(train_df)
    dataset_train,_=preprocessingForBert(train_data,tokenizer)

    dev_data=loadData(dev_df)
    dataset_val,lenValInp=preprocessingForBert(dev_data,tokenizer)

    #sampler=None, batch_size=bs, shuffle=False)
    dataloader_train=DataLoader(dataset_train,sampler=None, batch_size=batch_size, shuffle=False)#, batch_size=batch_size, worker_init_fn=seed_worker, generator=g)
    dataloader_val=DataLoader(dataset_val, batch_size=batch_size,sampler=None,shuffle=False)#worker_init_fn=seed_worker, generator=g )

    model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=3,
                                                          output_attentions=False,output_hidden_states=False)
    model.cuda()

    # criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    total_steps = len(dataloader_train) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_Micro,best_Macro=0.0,0.0
    for i in range(epochs):
        print('\n------training for epoch {}-----'.format(i+1))

        total_loss=0
        model.train()


        for step, batch in enumerate(dataloader_train):

            b_input_ids = batch[0].to(device)
            # b_token_type_id = batch[1].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device, dtype=torch.int64)

            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_attention_mask,labels=b_labels)

            # loss=criterion(outputs,b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()

            # optimizer.zero_grad()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(dataloader_train)
        print('average training loss: {}'.format(avg_train_loss))

        print('Running model validation')
        model.eval()
        predictions,true_scores=[],[]
        for batch in dataloader_val:
            batch=tuple(t.to(device) for t in batch)
            b_input_ids, b_attention_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids,attention_mask=b_attention_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.append(logits)
            true_scores.append(label_ids)

        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        

        #cols sentence, labels,  file_name, ann_type, drug, start, end
        tag_values = ['Disposition', 'NoDisposition', 'Undetermined']
        tag2idx = {i: t for i, t in enumerate(tag_values)}

        # flat_true_labels=[tag2idx[i] for i in flat_true_labels]
        flat_predictions=[tag2idx[i] for i in flat_predictions]

        # save the predictions to csv file for error analysis
        results = pd.DataFrame()
        results['sentence']=np.array([instance['sentence'] for instance in dev_data])
        results['true_label'] = np.array([tag2idx[instance['label']] for instance in dev_data])
        results['pred_label']=np.array(flat_predictions)
        results['drug'] = np.array([instance['drug'] for instance in dev_data])
        results['span_start'] = np.array([instance['span_start'] for instance in dev_data])
        results['span_end'] = np.array([instance['span_end'] for instance in dev_data])
        results['anno_type'] = np.array([instance['anno_type'] for instance in dev_data])
        results['filename'] = np.array([instance['filename'] for instance in dev_data])


        results_directory='../n2c2_models_results/{}'.format(model_name.split('/')[-1])
        try:
            os.mkdir(results_directory)
        except: pass

        micro, macro=evaluation(results,epoch='{}_{}'.format(str(i),str(learning_rate)),
                                resultsFile=os.path.join(results_directory,'{}_evalResults.txt'.format(type)))

        # save the model and predictions with the best micro F1 and macro F1 scores
        if micro>best_Micro:
            best_Micro=micro
            print('{}  best micro: {}'.format(type,best_Micro))
            results.to_csv(os.path.join(results_directory,'{}_bestMicro_results.csv'.format(type)),index=False)
            # results.to_csv('../metaAnalysisModels_Results/best_results/{}_bestMicro_results.csv'.format(type),index=False)
            torch.save(model, os.path.join(results_directory, 'best_micro_model_{}.pt'.format(type)))
        if macro>best_Macro:
            best_Macro=macro
            print('{}  best macro: {}'.format(type,best_Macro))
            results.to_csv(os.path.join(results_directory,'{}_bestMacro_results.csv'.format(type)),index=False)
            torch.save(model, os.path.join(results_directory, 'best_macro_model_{}.pt'.format(type)))


    return 0


def evaluation(forecasted_df,epoch,resultsFile):
    # evaluation is done using evaluation script provided by  the task organizers
    # in the training, all the predictions are saved in one csv file
    # to use the evaluation script provided by the n2c2 task organizers, we need to change predicted csv file to .ann files
    # we save predictions for each file in a different '.ann' file
    saveFolder = '../n2c2_models_results/predicted_ann_files'  #folder for saving predicted '.ann files'

    try:shutil.rmtree(saveFolder)
    except:pass
    try:os.mkdir(saveFolder)
    except:pass

    for i in range(len(forecasted_df)):
        filename = forecasted_df['filename'][i]
        save_file = open(os.path.join(saveFolder, filename), 'a')
        # format: # 'T1\tNoDisposition 821 827\tProzac\n'
        saveText1 = '{}\t{} {} {}\t{}\n'.format(forecasted_df['anno_type'][i],
                                                forecasted_df['pred_label'][i],
                                                forecasted_df['span_start'][i],
                                                forecasted_df['span_end'][i],
                                                forecasted_df['drug'][i])
        # E2\tDisposition:T2 \n
        saveText2 = '{}\t{}:{} \n'.format(forecasted_df['anno_type'][i].replace('T', 'E'),
                                          forecasted_df['pred_label'][i], forecasted_df['anno_type'][i])
        save_file.write(saveText1)
        save_file.write(saveText2)

    f1 = '../dev'  #path for folder with gold '.ann' dev files
    f2 = saveFolder  #path for folder with predicted '.ann' files
    n2c2_eval_script.main(f1, f2, verbose=False,epoch=epoch,resultsFile=resultsFile)

    #get micro and macro values
    readResultsFile=open(resultsFile,'r')
    lines = [l for l in readResultsFile]
    lines = lines[-2:]
    micro = lines[0].split('\t')[-1]
    macro = lines[1].split('\t')[-1]
    return float(micro), float(macro)







train_df=pd.read_csv('../train_sequences_window_200.csv', names=['sentence', 'label', 'filename', 'ann_type','drugname', 'start', 'end'])
dev_df=pd.read_csv('../dev_sequences_window_200.csv', names=['sentence', 'label', 'filename', 'ann_type','drugname', 'start', 'end'])
train_df.sample(frac=1)  #shuffle
dev_df.sample(frac=1)       #shuffle

def clinical_longF():
    model_name= 'yikuan8/Clinical-Longformer'

    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    print('{:*^50}'.format('TRAINING FOR MODEL ' + model_name))

    training(epochs=10, learning_rate=1e-5, train_df=train_df, dev_df=dev_df, type='window200',model_name=model_name, tokenizer=tokenizer,batch_size=20)


    return 0
