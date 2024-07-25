import torch
import numpy as np
import sys
sys.path.append('..')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def process_input(seq):
    """
    Separate the input sequence with spaces   (eg. "ABCD"--->"A B C D")
    param: seq: input sequence
    return: Sequence separated by spaces
    """
    pro_seq = ''
    for i in range(len(seq)):
        if i == 0:
            pro_seq += seq[i]
        else:
            pro_seq += " " + seq[i]
    return pro_seq

def get_tokenizer():
    import os
    from transformers import BertTokenizer
    project_dir = os.path.abspath('.')

    tokenizer_config_PATH = project_dir + "/pretrained_model_dir/tapebert"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_config_PATH, do_lower_case=False)
    return tokenizer


def get_MFBP_model(model_num):
    import os
    import torch
    from mfbp_model.BertForMultiLabelWithGAT_MFBP import BertForMultiLabelSequenceClassification

    project_dir = os.path.abspath('.')
    MFBP_model_PATH = project_dir  + f"/weights_mfbp/model_Peptide_MFBP{model_num}.bin"
    MFBP_config_PATH = project_dir + "/pretrained_model_dir/tapebert"
    finetune_model = BertForMultiLabelSequenceClassification(MFBP_config_PATH) 

    if os.path.exists(MFBP_model_PATH):
        loaded_paras = torch.load(MFBP_model_PATH, map_location='cpu')
        finetune_model.load_state_dict(loaded_paras, strict=False)
        print(f"## Successfully loaded {MFBP_model_PATH} model for inference ......")
    else:
        print("Model not found")

    return finetune_model.to(device)


def get_seq_pred_MFBP(seq, model, tokenizer, type):
    num_label = 5
        
    process_seq = process_input(seq)
    inputs = tokenizer.encode(process_seq, return_tensors='pt').to(device) 
    graph_info = np.ones((num_label,num_label))
    x = np.diag([1 for _ in range(num_label)])
    graph_info = graph_info - x
    adj_matrix = torch.from_numpy(graph_info).float()
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(
            input_ids=inputs,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            adj_matrix=None)
    # print(logits)
    logits = logits.sigmoid().cpu().numpy()
    logits = logits

    return logits


def pred_MFBP(seq, MFBP_model_list, MFBP_tokenizer):
    label_list = ["ACP", "ADP", "AHP", "AIP", "AMP"]
        
    # 10个模型预测取平均
    for i, model in enumerate(MFBP_model_list): 
        pred_res = get_seq_pred_MFBP(seq, model, MFBP_tokenizer, 'MFBP')
#         print(pred_res)
        if i == 0:
            pred_res_allmodel = pred_res[0]
        else:
            pred_res_allmodel = np.vstack((pred_res_allmodel, pred_res[0]))

    end_pred_res = pred_res_allmodel.mean(axis=0)
    end_pred_res_score = end_pred_res
    end_pred_res = end_pred_res > 0.5
    predict_labels = []
    for i, res in enumerate(end_pred_res):
        if res == True:
            predict_labels.append(label_list[i])
            
    return end_pred_res_score, predict_labels, label_list

def get_MFTP_model(model_num):
    import os
    import torch
    from mftp_model.BertForMultiLabelWithGAT_MFTP import BertForMultiLabelSequenceClassification

    project_dir = os.path.abspath('.')
    MFTP_model_PATH = project_dir  + f"/weights_mftp/model_Peptide_MFTP{model_num}.bin"
    MFTP_config_PATH = project_dir + "/pretrained_model_dir/tapebert"
    finetune_model = BertForMultiLabelSequenceClassification(MFTP_config_PATH) 

    if os.path.exists(MFTP_model_PATH):
        loaded_paras = torch.load(MFTP_model_PATH, map_location='cpu')
        finetune_model.load_state_dict(loaded_paras, strict=False)
        print(f"## Successfully loaded {MFTP_model_PATH} model for inference ......")
    else:
        print("Model not found")

    return finetune_model.to(device)


def get_seq_pred_MFTP(seq, model, tokenizer, type):
    num_label = 21
        
    process_seq = process_input(seq)
    inputs = tokenizer.encode(process_seq, return_tensors='pt').to(device) 
    graph_info = np.ones((num_label,num_label))
    x = np.diag([1 for _ in range(num_label)])
    graph_info = graph_info - x
    adj_matrix = torch.from_numpy(graph_info).float()
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(
            input_ids=inputs,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            adj_matrix=None)
    # print(logits)
    logits = logits.sigmoid().cpu().numpy()
    logits = logits

    return logits


def pred_MFTP(seq, MFTP_model_list, MFTP_tokenizer):
    label_list = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']    

    for i, model in enumerate(MFTP_model_list): 
        pred_res = get_seq_pred_MFTP(seq, model, MFTP_tokenizer, 'MFTP')
        if i == 0:
            pred_res_allmodel = pred_res[0]
        else:
            pred_res_allmodel = np.vstack((pred_res_allmodel, pred_res[0]))

    end_pred_res = pred_res_allmodel.mean(axis=0)
    end_pred_res_score = end_pred_res
    end_pred_res = end_pred_res > 0.5
    predict_labels = []
    for i, res in enumerate(end_pred_res):
        if res == True:
            predict_labels.append(label_list[i])
            
    return end_pred_res_score, predict_labels, label_list

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, default = None, help='Sequence to be predicted')
    parser.add_argument('--model-type', type=str, default = None, help='5-class, or 21-class')
    args = parser.parse_args()
    if args.seq == None or args.model_type == None:
        print("Please enter the sequence to be predicted and choose a model, for example: FGLPMLSILPKALCILLKRKC, 5-class")
    else:
        tokenizer = get_tokenizer()
        if args.model_type == '5-class':
            MFBP_model_list = list()
            for i in range(10):
                MFBP_model = get_MFBP_model(i)
                MFBP_model_list.append(MFBP_model)
            end_pred_res_scores, predict_labels, label_list = pred_MFBP(args.seq, MFBP_model_list, tokenizer)
        elif args.model_type == '21-class':
            MFTP_model_list = list()
            for i in range(10):
                MFTP_model = get_MFTP_model(i)
                MFTP_model_list.append(MFTP_model)
            end_pred_res_scores, predict_labels, label_list = pred_MFTP(args.seq, MFTP_model_list, tokenizer)
        else:
            print("Please choose a model type: 5-class or 21-class.")

        print('The existing labels: ', label_list)
        print('The predicted scores: ', end_pred_res_scores)
        print('The predicted labels: ', predict_labels)


### running command:
'''
CUDA_VISIBLE_DEVICES='' python predict.py --seq FGLPMLSILPKALCILLKRKC --model-type 5-class

CUDA_VISIBLE_DEVICES='' python predict.py --seq ACDTATCVTHRLAGLLSRSGGVVKNNFVPTNVGSKAF --model-type 21-class
'''