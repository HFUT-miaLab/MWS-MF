# STL
import copy
import os
import datetime
import argparse

# 3rd party library
import shutil
import sys

from MultiLabel_model import MultiLabel
from radam import RAdam
from utils import Logger
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
# local library
from MultiLabel_dataset import Teran_feat_label_Dataset, get_train_valid_names, get_test_names
import metric



class BestModelSaver:
    def __init__(self, max_epoch, ratio=0.3):
        self.best_valid_acc_AFB = 0
        self.best_valid_auc_AFB = 0
        self.best_valid_acc_epoch_AFB = 0
        self.best_valid_auc_epoch_AFB = 0
        self.best_valid_acc_GMS = 0
        self.best_valid_auc_GMS = 0
        self.best_valid_acc_epoch_GMS = 0
        self.best_valid_auc_epoch_GMS = 0
        self.best_valid_acc_PAS = 0
        self.best_valid_auc_PAS = 0
        self.best_valid_acc_epoch_PAS = 0
        self.best_valid_auc_epoch_PAS = 0

        self.begin_epoch = int(max_epoch * ratio)

    def update(self, valid_acc_AFB, valid_auc_AFB, valid_acc_GMS, valid_auc_GMS, valid_acc_PAS, valid_auc_PAS, current_epoch):
        if current_epoch < self.begin_epoch:
            return

        if valid_acc_AFB >= self.best_valid_acc_AFB:
            self.best_valid_acc_AFB = valid_acc_AFB
            self.best_valid_acc_epoch_AFB = current_epoch
        if valid_auc_AFB >= self.best_valid_auc_AFB:
            self.best_valid_auc_AFB = valid_auc_AFB
            self.best_valid_auc_epoch_AFB = current_epoch

        if valid_acc_GMS >= self.best_valid_acc_GMS:
            self.best_valid_acc_GMS = valid_acc_GMS
            self.best_valid_acc_epoch_GMS = current_epoch
        if valid_auc_GMS >= self.best_valid_auc_GMS:
            self.best_valid_auc_GMS = valid_auc_GMS
            self.best_valid_auc_epoch_GMS = current_epoch

        if valid_acc_PAS >= self.best_valid_acc_PAS:
            self.best_valid_acc_PAS = valid_acc_PAS
            self.best_valid_acc_epoch_PAS = current_epoch
        if valid_auc_PAS >= self.best_valid_auc_PAS:
            self.best_valid_auc_PAS = valid_auc_PAS
            self.best_valid_auc_epoch_PAS = current_epoch


def _macro_auc(lbl_true_list, lbl_pred_list, multi_class=False, n_classes=None):
    if multi_class:
        lbl_true = np.concatenate(lbl_true_list, axis=0)
        lbl_pred = np.concatenate(lbl_pred_list, axis=0)
        return roc_auc_score(lbl_true, lbl_pred, average="macro", multi_class='ovo')
    else:
        lbl_true = np.concatenate(lbl_true_list, axis=0)
        lbl_pred = np.concatenate(lbl_pred_list, axis=0)[:, 1]
        return roc_auc_score(lbl_true, lbl_pred, average="macro")


def _micro_auc(lbl_true_list, lbl_pred_list, multi_class=False, n_classes=None):
    if multi_class:
        lbl_true = np.concatenate(lbl_true_list, axis=0)
        lbl_pred = np.concatenate(lbl_pred_list, axis=0)
        return roc_auc_score(lbl_true, lbl_pred, average="micro", multi_class='ovr')
    else:
        lbl_true = np.concatenate(lbl_true_list, axis=0)
        lbl_pred = np.concatenate(lbl_pred_list, axis=0)[:, 1]
        return roc_auc_score(lbl_true, lbl_pred, average="micro")


def micro_auc(lbl_true_list, lbl_pred_list, multi_class=False, n_classes=None):
    lbl_true = np.concatenate(lbl_true_list, axis=0)
    lbl_pred = np.concatenate(lbl_pred_list, axis=0)

    if multi_class:
        aucs = []
        binary_labels = label_binarize(lbl_true, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in lbl_true:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], lbl_pred[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        return np.nanmean(np.array(aucs))
    else:
        return roc_auc_score(lbl_true, lbl_pred)


def eval(args, test_loader, model, eval_metric='ACC'):
    model.eval()

    correct_AFB, correct_GMS, correct_PAS, total = 0, 0, 0, 0

    lbl_true_list_AFB, lbl_pred_list_AFB = [], []
    lbl_true_list_GMS, lbl_pred_list_GMS = [], []
    lbl_true_list_PAS, lbl_pred_list_PAS = [], []

    y_result_AFB, pred_result_AFB, pred_probs_AFB = [], [], []
    y_result_GMS, pred_result_GMS, pred_probs_GMS = [], [], []
    y_result_PAS, pred_result_PAS, pred_probs_PAS = [], [], []

    site_results = {
        "site1": {
            "y_true_AFB": [], "y_pred_AFB": [], "y_prob_AFB": [],
            "y_true_GMS": [], "y_pred_GMS": [], "y_prob_GMS": [],
            "y_true_PAS": [], "y_pred_PAS": [], "y_prob_PAS": [],
        },
        "site2": {
            "y_true_AFB": [], "y_pred_AFB": [], "y_prob_AFB": [],
            "y_true_GMS": [], "y_pred_GMS": [], "y_prob_GMS": [],
            "y_true_PAS": [], "y_pred_PAS": [], "y_prob_PAS": [],
        }
    }

    with torch.no_grad():
        for step, (x, text, AFB_target, GMS_target, PAS_target, site) in enumerate(test_loader):
            x = x.squeeze()
            if torch.cuda.is_available():
                x = x.cuda(args.cuda_choice)
                text = text.cuda(args.cuda_choice)
                AFB_target = AFB_target.cuda(args.cuda_choice)
                GMS_target = GMS_target.cuda(args.cuda_choice)
                PAS_target = PAS_target.cuda(args.cuda_choice)
                site = site.cuda(args.cuda_choice)

            results_dict = model(x, text, AFB_target, GMS_target, PAS_target)
            logits_AFB, logits_GMS, logits_PAS = results_dict['logits_AFB'], results_dict['logits_GMS'], results_dict['logits_PAS']

            probs_AFB, probs_GMS, probs_PAS = torch.softmax(logits_AFB, -1), torch.softmax(logits_GMS, -1), torch.softmax(logits_PAS, -1)
            pred_AFB, pred_GMS, pred_PAS = torch.argmax(logits_AFB, -1), torch.argmax(logits_GMS, -1), torch.argmax(logits_PAS, -1)

            correct_AFB += int((pred_AFB == AFB_target).sum().cpu())
            correct_GMS += int((pred_GMS == GMS_target).sum().cpu())
            correct_PAS += int((pred_PAS == PAS_target).sum().cpu())

            # 全局结果存储
            y_result_AFB.append(AFB_target.cpu().numpy())
            pred_result_AFB += pred_AFB.tolist()
            pred_probs_AFB += probs_AFB.tolist()

            y_result_GMS.append(GMS_target.cpu().numpy())
            pred_result_GMS += pred_GMS.tolist()
            pred_probs_GMS += probs_GMS.tolist()

            y_result_PAS.append(PAS_target.cpu().numpy())
            pred_result_PAS += pred_PAS.tolist()
            pred_probs_PAS += probs_PAS.tolist()

            lbl_true_list_AFB.append(AFB_target.cpu().numpy())
            lbl_pred_list_AFB.append(probs_AFB.cpu().numpy())
            lbl_true_list_GMS.append(GMS_target.cpu().numpy())
            lbl_pred_list_GMS.append(probs_GMS.cpu().numpy())
            lbl_true_list_PAS.append(PAS_target.cpu().numpy())
            lbl_pred_list_PAS.append(probs_PAS.cpu().numpy())

            # site 分组结果存储
            for i in range(len(site)):
                s = "site"+str(site[i].item())
                site_results[s]["y_true_AFB"].append(int(AFB_target[i].cpu().item()))
                site_results[s]["y_pred_AFB"].append(int(pred_AFB[i].cpu().item()))
                site_results[s]["y_prob_AFB"].append(probs_AFB[i].cpu().numpy())

                site_results[s]["y_true_GMS"].append(int(GMS_target[i].cpu().item()))
                site_results[s]["y_pred_GMS"].append(int(pred_GMS[i].cpu().item()))
                site_results[s]["y_prob_GMS"].append(probs_GMS[i].cpu().numpy())

                site_results[s]["y_true_PAS"].append(int(PAS_target[i].cpu().item()))
                site_results[s]["y_pred_PAS"].append(int(pred_PAS[i].cpu().item()))
                site_results[s]["y_prob_PAS"].append(probs_PAS[i].cpu().numpy())

            total += len(AFB_target)

    acc_AFB, acc_GMS, acc_PAS = correct_AFB / total, correct_GMS / total, correct_PAS / total

    if args.nclass == 2:
        class_names = ['negative', 'positive']
    else:
        raise NotImplementedError
    
    metric.draw_confusion_matrix(y_result_AFB, pred_result_AFB, class_names=class_names,
                             save_path=os.path.join(args.fold_save_path, 'Total_AFB_Confusion_Matrix_' + eval_metric + '.jpg'))
    if args.nclass == 2:
        metric.draw_binary_roc_curve(y_result_AFB, np.array(pred_probs_AFB)[:, 1],save_path=os.path.join(args.fold_save_path,
                                                        'Total_AFB_ROC_Curve_' + eval_metric + '.jpg') , name = "AFB")
    else:
        metric.draw_muti_roc_curve(y_result_AFB, np.array(pred_probs_AFB), class_names=class_names,save_path=os.path.join(args.fold_save_path,
                                                          'Total_AFB_ROC_Curve_' + eval_metric + '.jpg') , name = "AFB")
    
    metric.draw_confusion_matrix(y_result_GMS, pred_result_GMS, class_names=class_names,
                             save_path=os.path.join(args.fold_save_path, 'Total_GMS_Confusion_Matrix_' + eval_metric + '.jpg'))
    if args.nclass == 2:
        metric.draw_binary_roc_curve(y_result_GMS, np.array(pred_probs_GMS)[:, 1],save_path=os.path.join(args.fold_save_path,
                                                            'Total_GMS_ROC_Curve_' + eval_metric + '.jpg'), name = "GMS")
    else:
        metric.draw_muti_roc_curve(y_result_GMS, np.array(pred_probs_GMS), class_names=class_names,save_path=os.path.join(args.fold_save_path,
                                                          'Total_GMS_ROC_Curve_' + eval_metric + '.jpg'), name = "GMS")
    
    metric.draw_confusion_matrix(y_result_PAS, pred_result_PAS, class_names=class_names,
                             save_path=os.path.join(args.fold_save_path,'Total_PAS_Confusion_Matrix_' + eval_metric + '.jpg'))
    if args.nclass == 2:
        metric.draw_binary_roc_curve(y_result_PAS, np.array(pred_probs_PAS)[:, 1],save_path=os.path.join(args.fold_save_path,
                                                            'Total_PAS_ROC_Curve_' + eval_metric + '.jpg'), name = "PAS")
    else:
        metric.draw_muti_roc_curve(y_result_PAS, np.array(pred_probs_PAS), class_names=class_names,save_path=os.path.join(args.fold_save_path,
                                                          'Total_PAS_ROC_Curve_' + eval_metric + '.jpg'), name = "PAS")
        
    metric.draw_confusion_matrix(site_results["site1"]["y_true_AFB"], site_results["site1"]["y_pred_AFB"], class_names=class_names,
                                 save_path=os.path.join(args.fold_save_path,
                                                        'Lung_AFB_Confusion_Matrix_' + eval_metric + '.jpg'), name = "AFB (Lung)")
    if args.nclass == 2:
        metric.draw_binary_roc_curve(site_results["site1"]["y_true_AFB"], np.array(site_results["site1"]["y_prob_AFB"])[:, 1],
                                     save_path=os.path.join(args.fold_save_path,
                                                            'Lung_AFB_ROC_Curve_' + eval_metric + '.jpg'), name = "AFB (Lung)")
    else:
        metric.draw_muti_roc_curve(site_results["site1"]["y_true_AFB"], np.array(site_results["site1"]["y_prob_AFB"]), class_names=class_names,
                                   save_path=os.path.join(args.fold_save_path,
                                                          'Lung_AFB_ROC_Curve_' + eval_metric + '.jpg'), name = "AFB (Lung)")

    metric.draw_confusion_matrix(site_results["site1"]["y_true_GMS"], site_results["site1"]["y_pred_GMS"],
                                 class_names=class_names,
                                 save_path=os.path.join(args.fold_save_path,
                                                        'Lung_GMS_Confusion_Matrix_' + eval_metric + '.jpg'), name = "GMS (Lung)")
    if args.nclass == 2:
        metric.draw_binary_roc_curve(site_results["site1"]["y_true_GMS"],
                                     np.array(site_results["site1"]["y_prob_GMS"])[:, 1],
                                     save_path=os.path.join(args.fold_save_path,
                                                            'Lung_GMS_ROC_Curve_' + eval_metric + '.jpg'), name = "GMS (Lung)")
    else:
        metric.draw_muti_roc_curve(site_results["site1"]["y_true_GMS"], np.array(site_results["site1"]["y_prob_GMS"]),
                                   class_names=class_names,
                                   save_path=os.path.join(args.fold_save_path,
                                                          'Lung_GMS_ROC_Curve_' + eval_metric + '.jpg'), name = "GMS (Lung)")

    metric.draw_confusion_matrix(site_results["site1"]["y_true_PAS"], site_results["site1"]["y_pred_PAS"],
                                 class_names=class_names,
                                 save_path=os.path.join(args.fold_save_path,
                                                        'Lung_PAS_Confusion_Matrix_' + eval_metric + '.jpg'), name = "PAS (Lung)")
    if args.nclass == 2:
        metric.draw_binary_roc_curve(site_results["site1"]["y_true_PAS"],
                                     np.array(site_results["site1"]["y_prob_PAS"])[:, 1],
                                     save_path=os.path.join(args.fold_save_path,
                                                            'Lung_PAS_ROC_Curve_' + eval_metric + '.jpg'), name = "PAS (Lung)")
    else:
        metric.draw_muti_roc_curve(site_results["site1"]["y_true_PAS"], np.array(site_results["site1"]["y_prob_PAS"]),
                                   class_names=class_names,
                                   save_path=os.path.join(args.fold_save_path,
                                                          'Lung_PAS_ROC_Curve_' + eval_metric + '.jpg'), name = "PAS (Lung)")
        

    metric.draw_confusion_matrix(site_results["site2"]["y_true_AFB"], site_results["site2"]["y_pred_AFB"],
                                 class_names=class_names,
                                 save_path=os.path.join(args.fold_save_path,
                                                        'Other_AFB_Confusion_Matrix_' + eval_metric + '.jpg'), name = "AFB (Other)")
    if args.nclass == 2:
        metric.draw_binary_roc_curve(site_results["site2"]["y_true_AFB"],
                                     np.array(site_results["site2"]["y_prob_AFB"])[:, 1],
                                     save_path=os.path.join(args.fold_save_path,
                                                            'Other_AFB_ROC_Curve_' + eval_metric + '.jpg'), name = "AFB (Other)")
    else:
        metric.draw_muti_roc_curve(site_results["site2"]["y_true_AFB"], np.array(site_results["site2"]["y_prob_AFB"]),
                                   class_names=class_names,
                                   save_path=os.path.join(args.fold_save_path,
                                                          'Other_AFB_ROC_Curve_' + eval_metric + '.jpg'), name = "AFB (Other)")

    metric.draw_confusion_matrix(site_results["site2"]["y_true_GMS"], site_results["site2"]["y_pred_GMS"],
                                 class_names=class_names,
                                 save_path=os.path.join(args.fold_save_path,
                                                        'Other_GMS_Confusion_Matrix_' + eval_metric + '.jpg'), name = "GMS (Other)")
    if args.nclass == 2:
        metric.draw_binary_roc_curve(site_results["site2"]["y_true_GMS"],
                                     np.array(site_results["site2"]["y_prob_GMS"])[:, 1],
                                     save_path=os.path.join(args.fold_save_path,
                                                            'Other_GMS_ROC_Curve_' + eval_metric + '.jpg'), name = "GMS (Other)")
    else:
        metric.draw_muti_roc_curve(site_results["site2"]["y_true_GMS"], np.array(site_results["site2"]["y_prob_GMS"]),
                                   class_names=class_names,
                                   save_path=os.path.join(args.fold_save_path,
                                                          'Other_GMS_ROC_Curve_' + eval_metric + '.jpg'), name = "GMS (Other)")

    metric.draw_confusion_matrix(site_results["site2"]["y_true_PAS"], site_results["site2"]["y_pred_PAS"],
                                 class_names=class_names,
                                 save_path=os.path.join(args.fold_save_path,
                                                        'Other_PAS_Confusion_Matrix_' + eval_metric + '.jpg'), name = "PAS (Other)")
    if args.nclass == 2:
        metric.draw_binary_roc_curve(site_results["site2"]["y_true_PAS"],
                                     np.array(site_results["site2"]["y_prob_PAS"])[:, 1],
                                     save_path=os.path.join(args.fold_save_path,
                                                            'Other_PAS_ROC_Curve_' + eval_metric + '.jpg'), name = "PAS (Other)")
    else:
        metric.draw_muti_roc_curve(site_results["site2"]["y_true_PAS"], np.array(site_results["site2"]["y_prob_PAS"]),
                                   class_names=class_names,
                                   save_path=os.path.join(args.fold_save_path,
                                                          'Other_PAS_ROC_Curve_' + eval_metric + '.jpg'), name = "PAS (Other)")

    macro_auc_score_AFB = _macro_auc(lbl_true_list_AFB, lbl_pred_list_AFB, multi_class=args.nclass > 2, n_classes=args.nclass)
    macro_auc_score_GMS = _macro_auc(lbl_true_list_GMS, lbl_pred_list_GMS, multi_class=args.nclass > 2, n_classes=args.nclass)
    macro_auc_score_PAS = _macro_auc(lbl_true_list_PAS, lbl_pred_list_PAS, multi_class=args.nclass > 2, n_classes=args.nclass)

    precision_AFB, recall_AFB, F1_score_AFB = precision_recall_fscore_support(y_result_AFB, pred_result_AFB, average='macro')[:-1]
    specificity_AFB = metric.compute_specificity(y_result_AFB, pred_result_AFB)
    sensitivity_AFB = metric.compute_sensitivity(y_result_AFB, pred_result_AFB)

    precision_GMS, recall_GMS, F1_score_GMS = precision_recall_fscore_support(y_result_GMS, pred_result_GMS, average='macro')[:-1]
    specificity_GMS = metric.compute_specificity(y_result_GMS, pred_result_GMS)
    sensitivity_GMS = metric.compute_sensitivity(y_result_GMS, pred_result_GMS)

    precision_PAS, recall_PAS, F1_score_PAS = precision_recall_fscore_support(y_result_PAS, pred_result_PAS, average='macro')[:-1]
    specificity_PAS = metric.compute_specificity(y_result_PAS, pred_result_PAS)
    sensitivity_PAS = metric.compute_sensitivity(y_result_PAS, pred_result_PAS)

    site_metrics = {}
    for s in site_results:
        site_metrics[s] = {}
        for stain in ["AFB", "GMS", "PAS"]:
            y_true = site_results[s][f"y_true_{stain}"]
            y_pred = site_results[s][f"y_pred_{stain}"]
            y_prob = site_results[s][f"y_prob_{stain}"]

            if len(y_true) == 0:  
                continue

            auc = roc_auc_score(y_true, np.array(y_prob)[:, 1]) if args.nclass == 2 else _macro_auc([y_true], [y_prob], multi_class=True, n_classes=args.nclass)
            precision, recall, f1 = precision_recall_fscore_support(y_true, y_pred, average="macro")[:-1]
            specificity = metric.compute_specificity(y_true, y_pred)
            sensitivity = metric.compute_sensitivity(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)

            site_metrics[s][stain] = {
                "acc": acc,
                "macro_auc": auc,
                "precision": precision,
                "recall": recall,
                "F1": f1,
                "specificity": specificity,
                "sensitivity": sensitivity,
            }

    # ===== 返回 =====

    return acc_AFB, macro_auc_score_AFB, precision_AFB, recall_AFB, F1_score_AFB, specificity_AFB, sensitivity_AFB\
            ,acc_GMS, macro_auc_score_GMS, precision_GMS, recall_GMS, F1_score_GMS, specificity_GMS, sensitivity_GMS\
            ,acc_PAS, macro_auc_score_PAS, precision_PAS, recall_PAS, F1_score_PAS, specificity_PAS, sensitivity_PAS\
            , site_metrics



def valid(args, valid_loader, model):
    model.eval()

    correct_AFB = 0
    correct_GMS = 0
    correct_PAS = 0

    total = 0
    gts_AFB = []
    preds_AFB = []
    gts_GMS = []
    preds_GMS = []
    gts_PAS = []
    preds_PAS = []

    with torch.no_grad():
        for step, (x, text, AFB_target, GMS_target, PAS_target, site) in enumerate(valid_loader):
            x = x.squeeze()
            if torch.cuda.is_available():
                x = x.cuda(args.cuda_choice)
                text = text.cuda(args.cuda_choice)
                AFB_target = AFB_target.cuda(args.cuda_choice)
                GMS_target = GMS_target.cuda(args.cuda_choice)
                PAS_target = PAS_target.cuda(args.cuda_choice)

            results_dict = model(x,text, AFB_target, GMS_target, PAS_target)  # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
            logits_AFB = results_dict['logits_AFB']
            probs_AFB = torch.softmax(logits_AFB, dim=-1)
            pred_AFB = torch.argmax(logits_AFB, dim=-1)

            logits_GMS = results_dict['logits_GMS']
            probs_GMS = torch.softmax(logits_GMS, dim=-1)
            pred_GMS = torch.argmax(logits_GMS, dim=-1)

            logits_PAS = results_dict['logits_PAS']
            probs_PAS = torch.softmax(logits_PAS, dim=-1)
            pred_PAS = torch.argmax(logits_PAS, dim=-1)

            # pred = torch.cat([pred_AFB, pred_GMS, pred_PAS])

            # correct = correct + int((pred == target).sum().cpu())
            correct_AFB += int((pred_AFB == AFB_target).sum().cpu())
            correct_GMS += int((pred_GMS == GMS_target).sum().cpu())
            correct_PAS += int((pred_PAS == PAS_target).sum().cpu())

            gts_AFB.append(AFB_target.detach().cpu().numpy())
            preds_AFB.append(probs_AFB.detach().cpu().numpy())
            gts_GMS.append(GMS_target.detach().cpu().numpy())
            preds_GMS.append(probs_GMS.detach().cpu().numpy())
            gts_PAS.append(PAS_target.detach().cpu().numpy())
            preds_PAS.append(probs_PAS.detach().cpu().numpy())

            total += len(PAS_target)

    acc_AFB = correct_AFB / total
    acc_GMS = correct_GMS / total
    acc_PAS = correct_PAS / total

    auc_score_AFB = _macro_auc(gts_AFB, preds_AFB, multi_class=args.nclass > 2, n_classes=args.nclass)
    auc_score_GMS = _macro_auc(gts_GMS, preds_GMS, multi_class=args.nclass > 2, n_classes=args.nclass)
    auc_score_PAS = _macro_auc(gts_PAS, preds_PAS, multi_class=args.nclass > 2, n_classes=args.nclass)

    return acc_AFB, auc_score_AFB, acc_GMS, auc_score_GMS, acc_PAS, auc_score_PAS

def compute_contrastive_loss( logits, queue_len=5):
    contra_loss = contrastive_loss(logits, queue_len)
    return contra_loss

def contrastive_loss(logits: torch.Tensor, queue_len=5) -> torch.Tensor:
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(logits.unsqueeze(0), torch.tensor([queue_len - 1], device=logits.device))

def train(args, model, train_loader, valid_loader):
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0)

    args.current_epoch = 0
    best_model_saver = BestModelSaver(args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        args.current_lr = lr

        train_loss = 0
        train_AFB_loss = 0
        train_GMS_loss = 0
        train_PAS_loss = 0
        # correct = 0
        correct_AFB = 0
        correct_GMS = 0
        correct_PAS = 0
        total = 0

        total_step = 0

        for step, (x, text ,AFB_target, GMS_target, PAS_target, site) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.squeeze()

            if torch.cuda.is_available():
                x = x.cuda(args.cuda_choice)
                text = text.cuda(args.cuda_choice)
                AFB_target = AFB_target.cuda(args.cuda_choice)
                GMS_target = GMS_target.cuda(args.cuda_choice)
                PAS_target = PAS_target.cuda(args.cuda_choice)

            results_dict = model(x, text, AFB_target, GMS_target, PAS_target)   
            logits_AFB = results_dict['logits_AFB']
            probs_AFB = torch.softmax(logits_AFB, dim=-1)
            pred_AFB = torch.argmax(logits_AFB, dim=-1)

            logits_GMS = results_dict['logits_GMS']
            probs_GMS = torch.softmax(logits_GMS, dim=-1)
            pred_GMS = torch.argmax(logits_GMS, dim=-1)

            logits_PAS = results_dict['logits_PAS']
            probs_PAS = torch.softmax(logits_PAS, dim=-1)
            pred_PAS = torch.argmax(logits_PAS, dim=-1)

            correct_AFB += int((pred_AFB == AFB_target).sum().cpu())
            correct_GMS += int((pred_GMS == GMS_target).sum().cpu())
            correct_PAS += int((pred_PAS == PAS_target).sum().cpu())
            total += 1
            loss_AFB = results_dict['loss_AFB']
            loss_GMS = results_dict['loss_GMS']
            loss_PAS = results_dict['loss_PAS']
            loss1 = (loss_AFB + loss_GMS + loss_PAS) / 3.0
            loss2 = (results_dict['L_kl_AFB'] + results_dict['L_kl_GMS'] + results_dict['L_kl_AFB']) / 3.0

            loss = loss1 + loss2 + results_dict['L_cos']
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_AFB_loss += loss_AFB
            train_GMS_loss += loss_GMS
            train_PAS_loss += loss_PAS

            total_step += 1
            if step % 50 == 0:
                print("\tEpoch: [{}/{}] || epochiter: [{}/{}] || Train Total Loss: {:.6f} || AFB Loss: {:.6f} || GMS Loss: {:.6f} || PAS Loss: {:.6f} || LR: {:.6f}"
                      .format(args.current_epoch + 1, args.epochs, step + 1, len(train_loader),
                              train_loss / total_step, train_AFB_loss / total_step, train_GMS_loss / total_step, train_PAS_loss / total_step, args.current_lr))

        train_acc_AFB = correct_AFB / total
        train_acc_GMS = correct_GMS / total
        train_acc_PAS = correct_PAS / total

        valid_acc_AFB, valid_auc_AFB, valid_acc_GMS, valid_auc_GMS, valid_acc_PAS, valid_auc_PAS = valid(args, valid_loader, model)
        best_model_saver.update(valid_acc_AFB, valid_auc_AFB, valid_acc_GMS, valid_auc_GMS, valid_acc_PAS, valid_auc_PAS, args.current_epoch)
        print('\tValidation-Epoch: {} || train_acc_AFB: {:.6f} || train_avg_loss_AFB: {:.6f} || valid_acc_AFB: {:.6f} || valid_auc_AFB: {:.6f}\n' 
            '\t train_acc_GMS: {:.6f} || train_avg_loss_GMS: {:.6f} || valid_acc_GMS: {:.6f} || valid_auc_GMS: {:.6f}\n'
              '\t train_acc_PAS: {:.6f} || train_avg_loss_PAS: {:.6f} || valid_acc_PAS: {:.6f} || valid_auc_PAS: {:.6f}\n'
              .format(args.current_epoch + 1, train_acc_AFB, train_AFB_loss / total_step, valid_acc_AFB, valid_auc_AFB,
                      train_acc_GMS, train_GMS_loss / total_step, valid_acc_GMS, valid_auc_GMS,
                      train_acc_PAS, train_PAS_loss / total_step, valid_acc_PAS, valid_auc_PAS))

        current_model_weight = copy.deepcopy(model.state_dict())
        torch.save(current_model_weight,os.path.join(args.fold_save_path, 'epoch' + str(args.current_epoch) + '.pth'))
        args.current_epoch += 1

    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_acc_epoch_AFB) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_acc_AFB.pth'))
    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_auc_epoch_AFB) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_auc_AFB.pth'))
    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_acc_epoch_GMS) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_acc_GMS.pth'))
    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_auc_epoch_GMS) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_auc_GMS.pth'))
    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_acc_epoch_PAS) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_acc_PAS.pth'))
    shutil.copyfile(os.path.join(args.fold_save_path, 'epoch' + str(best_model_saver.best_valid_auc_epoch_PAS) + '.pth'),
                    os.path.join(args.fold_save_path, 'best_auc_PAS.pth'))
    return best_model_saver


def init_results(models=("AFB", "GMS", "PAS"),
                 metrics=("acc", "macro_auc", "precision", "recall", "F1", "specificity", "sensitivity"),
                 stains=("AFB", "GMS", "PAS"),
                 best_types=("acc", "auc"),
                 sites=("total", "site1", "site2")):
    
    results = {}
    for model in models:  # AFB / GMS / PAS
        results[model] = {}
        for best in best_types:  # acc / auc
            key_best = f"{model}_{best}"  # e.g. AFB_acc
            results[model][best] = {}
            for stain in stains:  # AFB / GMS / PAS
                results[model][best][stain] = {}
                for metricname in metrics:  # acc / auc / recall ...
                    results[model][best][stain][metricname] = {}
                    for site in sites:  # tatal, site1, site2
                        results[model][best][stain][metricname][site] = []
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ours training script')
    #这里填入parser参数

    args = parser.parse_args()

    args.weights_save_path = os.path.join(args.save_path,
                                     datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f'))
    os.makedirs(args.weights_save_path, exist_ok=True)

    # 运行日志
    os.makedirs(args.logger_path, exist_ok=True)
    sys.stdout = Logger(
        filename=os.path.join(args.logger_path, datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt'))

    results = init_results()

    for fold in range(5):

        args.fold_save_path = os.path.join(args.weights_save_path, 'fold' + str(fold))
        os.makedirs(args.fold_save_path, exist_ok=True)

        print('Training Folder: {}.\n\tData Loading...'.format(fold))
        train_names, train_labels, train_sites, valid_names, valid_labels, valid_sites = get_train_valid_names(args.train_valid_csv,
                                                                                        fold=fold,
                                                                                        nclass=args.nclass)
        # sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights))
        train_dataset = Teran_feat_label_Dataset(args.features_root, args.text_features_root,train_names, train_labels, train_sites)
        valid_dataset = Teran_feat_label_Dataset(args.features_root, args.text_features_root,valid_names, valid_labels, valid_sites)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,pin_memory=True)

        model = MultiLabel(AFB_class = args.nclass, GMS_class = args.nclass, PAS_class = args.nclass).cuda()

        if torch.cuda.is_available():
            model = model.cuda(args.cuda_choice)
        best_model_saver = train(args, model, train_loader, valid_loader)
        print('\t(Valid)AFB Best ACC: {:.6f} at {} epoch || Best AUC: {:.6f} at {} epoch\n'
              '\t(Valid)GMS Best ACC: {:.6f} at {} epoch || Best AUC: {:.6f} at {} epoch\n'
              '\t(Valid)PAS Best ACC: {:.6f} at {} epoch || Best AUC: {:.6f} at {} epoch\n'
              .format(best_model_saver.best_valid_acc_AFB, best_model_saver.best_valid_acc_epoch_AFB, best_model_saver.best_valid_auc_AFB, best_model_saver.best_valid_auc_epoch_AFB
                    ,best_model_saver.best_valid_acc_GMS, best_model_saver.best_valid_acc_epoch_GMS, best_model_saver.best_valid_auc_GMS, best_model_saver.best_valid_auc_epoch_GMS
                    ,best_model_saver.best_valid_acc_PAS, best_model_saver.best_valid_acc_epoch_PAS, best_model_saver.best_valid_auc_PAS, best_model_saver.best_valid_auc_epoch_PAS))

        test_names, test_labels, test_sites = get_test_names(args.test_csv, nclass=args.nclass)
        test_dataset = Teran_feat_label_Dataset(args.features_root, args.text_features_root, test_names, test_labels, test_sites)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        best_acc_model_weight_AFB = torch.load(os.path.join(args.fold_save_path, 'best_acc_AFB.pth'))
        model.load_state_dict(best_acc_model_weight_AFB)

        (
            acc_AFB, macro_auc_score_AFB, precision_AFB, recall_AFB, F1_score_AFB, specificity_AFB, sensitivity_AFB,
            acc_GMS, macro_auc_score_GMS, precision_GMS, recall_GMS, F1_score_GMS, specificity_GMS, sensitivity_GMS,
            acc_PAS, macro_auc_score_PAS, precision_PAS, recall_PAS, F1_score_PAS, specificity_PAS, sensitivity_PAS,
            site_metrics
        ) = eval(args, test_loader, model, eval_metric='_ON_AFB_BEST_ACC_MODEL')

        # 所有染色和指标名
        stains = ["AFB", "GMS", "PAS"]
        metrics = ["acc", "macro_auc", "precision", "recall", "F1", "specificity", "sensitivity"]

        overall_values = {
            "AFB": (
            acc_AFB, macro_auc_score_AFB, precision_AFB, recall_AFB, F1_score_AFB, specificity_AFB, sensitivity_AFB),
            "GMS": (
            acc_GMS, macro_auc_score_GMS, precision_GMS, recall_GMS, F1_score_GMS, specificity_GMS, sensitivity_GMS),
            "PAS": (
            acc_PAS, macro_auc_score_PAS, precision_PAS, recall_PAS, F1_score_PAS, specificity_PAS, sensitivity_PAS),
        }

        for stain_eval, values in overall_values.items():
            for metric_name, value in zip(metrics, values):
                results["AFB"]["acc"][stain_eval][metric_name]["total"].append(value)

        for site, site_data in site_metrics.items():  # site = "site1", "site2"
            for stain_test, metrics_dict in site_data.items():  # stain_test = "AFB", "GMS", "PAS"
                for metric_name, value in metrics_dict.items():
                    results["AFB"]["acc"][stain_test][metric_name][site].append(value)

        print("Total:")
        for stain in stains:
            values = []
            for metric_name in metrics:
                val_list = results["AFB"]["acc"][stain][metric_name]["total"]
                values.append(val_list[-1] if val_list else float('nan'))
            print(f"\t(Test){stain} Best ACC model || " +
                  " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        for site in ["site1", "site2"]:
            print(f"\n{site.capitalize()}:")
            for stain in stains:
                values = []
                for metric_name in metrics:
                    val_list = results["AFB"]["acc"][stain][metric_name][site]
                    values.append(val_list[-1] if val_list else float('nan'))
                print(f"\t(Test){stain} Best ACC model || " +
                      " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        # ------------------- BEST AFB AUC MODEL TEST -------------------
        best_auc_model_weight_AFB = torch.load(os.path.join(args.fold_save_path, 'best_auc_AFB.pth'))
        model.load_state_dict(best_auc_model_weight_AFB)

        (
            acc_AFB, macro_auc_score_AFB, precision_AFB, recall_AFB, F1_score_AFB, specificity_AFB, sensitivity_AFB,
            acc_GMS, macro_auc_score_GMS, precision_GMS, recall_GMS, F1_score_GMS, specificity_GMS, sensitivity_GMS,
            acc_PAS, macro_auc_score_PAS, precision_PAS, recall_PAS, F1_score_PAS, specificity_PAS, sensitivity_PAS,
            site_metrics
        ) = eval(args, test_loader, model, eval_metric='_ON_AFB_BEST_AUC_MODEL')

        for stain_eval, values in overall_values.items():
            for metric_name, value in zip(metrics, values):
                results["AFB"]["auc"][stain_eval][metric_name]["total"].append(value)

        for site, site_data in site_metrics.items():  # site = "site1", "site2"
            for stain_test, metrics_dict in site_data.items():  # stain_test = "AFB", "GMS", "PAS"
                for metric_name, value in metrics_dict.items():
                    results["AFB"]["auc"][stain_test][metric_name][site].append(value)

        print("Total:")
        for stain in stains:
            values = []
            for metric_name in metrics:
                val_list = results["AFB"]["auc"][stain][metric_name]["total"]
                values.append(val_list[-1] if val_list else float('nan'))
            print(f"\t(Test){stain} Best AUC model || " +
                  " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        for site in ["site1", "site2"]:
            print(f"\n{site.capitalize()}:")
            for stain in stains:
                values = []
                for metric_name in metrics:
                    val_list = results["AFB"]["auc"][stain][metric_name][site]
                    values.append(val_list[-1] if val_list else float('nan'))
                print(f"\t(Test){stain} Best AUC model || " +
                      " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        # ------------------- BEST GMS ACC MODEL TEST -------------------
        best_acc_model_weight_GMS = torch.load(os.path.join(args.fold_save_path, 'best_acc_GMS.pth'))
        model.load_state_dict(best_acc_model_weight_GMS)

        (
            acc_AFB, macro_auc_score_AFB, precision_AFB, recall_AFB, F1_score_AFB, specificity_AFB,
            sensitivity_AFB,
            acc_GMS, macro_auc_score_GMS, precision_GMS, recall_GMS, F1_score_GMS, specificity_GMS,
            sensitivity_GMS,
            acc_PAS, macro_auc_score_PAS, precision_PAS, recall_PAS, F1_score_PAS, specificity_PAS,
            sensitivity_PAS,
            site_metrics
        ) = eval(args, test_loader, model, eval_metric='_ON_GMS_BEST_ACC_MODEL')

        for stain_eval, values in overall_values.items():
            for metric_name, value in zip(metrics, values):
                results["GMS"]["acc"][stain_eval][metric_name]["total"].append(value)

        for site, site_data in site_metrics.items():  # site = "site1", "site2"
            for stain_test, metrics_dict in site_data.items():  # stain_test = "AFB", "GMS", "PAS"
                for metric_name, value in metrics_dict.items():
                    results["GMS"]["acc"][stain_test][metric_name][site].append(value)

        print("Total:")
        for stain in stains:
            values = []
            for metric_name in metrics:
                val_list = results["GMS"]["acc"][stain][metric_name]["total"]
                values.append(val_list[-1] if val_list else float('nan'))
            print(f"\t(Test){stain} Best ACC model || " +
                  " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        for site in ["site1", "site2"]:
            print(f"\n{site.capitalize()}:")
            for stain in stains:
                values = []
                for metric_name in metrics:
                    val_list = results["GMS"]["acc"][stain][metric_name][site]
                    values.append(val_list[-1] if val_list else float('nan'))
                print(f"\t(Test){stain} Best ACC model || " +
                      " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        # ------------------- BEST GMS AUC MODEL TEST -------------------
        best_auc_model_weight_GMS = torch.load(os.path.join(args.fold_save_path, 'best_auc_GMS.pth'))
        model.load_state_dict(best_auc_model_weight_GMS)

        (
            acc_AFB, macro_auc_score_AFB, precision_AFB, recall_AFB, F1_score_AFB, specificity_AFB,
            sensitivity_AFB,
            acc_GMS, macro_auc_score_GMS, precision_GMS, recall_GMS, F1_score_GMS, specificity_GMS,
            sensitivity_GMS,
            acc_PAS, macro_auc_score_PAS, precision_PAS, recall_PAS, F1_score_PAS, specificity_PAS,
            sensitivity_PAS,
            site_metrics
        ) = eval(args, test_loader, model, eval_metric='_ON_GMS_BEST_AUC_MODEL')

        for stain_eval, values in overall_values.items():
            for metric_name, value in zip(metrics, values):
                results["GMS"]["auc"][stain_eval][metric_name]["total"].append(value)

        for site, site_data in site_metrics.items():  # site = "site1", "site2"
            for stain_test, metrics_dict in site_data.items():  # stain_test = "AFB", "GMS", "PAS"
                for metric_name, value in metrics_dict.items():
                    results["GMS"]["auc"][stain_test][metric_name][site].append(value)

        print("Total:")
        for stain in stains:
            values = []
            for metric_name in metrics:
                val_list = results["GMS"]["auc"][stain][metric_name]["total"]
                values.append(val_list[-1] if val_list else float('nan'))
            print(f"\t(Test){stain} Best AUC model || " +
                  " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        for site in ["site1", "site2"]:
            print(f"\n{site.capitalize()}:")
            for stain in stains:
                values = []
                for metric_name in metrics:
                    val_list = results["GMS"]["auc"][stain][metric_name][site]
                    values.append(val_list[-1] if val_list else float('nan'))
                print(f"\t(Test){stain} Best AUC model || " +
                      " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        best_acc_model_weight_PAS = torch.load(os.path.join(args.fold_save_path, 'best_acc_PAS.pth'))
        model.load_state_dict(best_acc_model_weight_PAS)

        (
            acc_AFB, macro_auc_score_AFB, precision_AFB, recall_AFB, F1_score_AFB, specificity_AFB,
            sensitivity_AFB,
            acc_GMS, macro_auc_score_GMS, precision_GMS, recall_GMS, F1_score_GMS, specificity_GMS,
            sensitivity_GMS,
            acc_PAS, macro_auc_score_PAS, precision_PAS, recall_PAS, F1_score_PAS, specificity_PAS,
            sensitivity_PAS,
            site_metrics
        ) = eval(args, test_loader, model, eval_metric='_ON_PAS_BEST_ACC_MODEL')

        for stain_eval, values in overall_values.items():
            for metric_name, value in zip(metrics, values):
                results["PAS"]["acc"][stain_eval][metric_name]["total"].append(value)

        for site, site_data in site_metrics.items():  # site = "site1", "site2"
            for stain_test, metrics_dict in site_data.items():  # stain_test = "AFB", "GMS", "PAS"
                for metric_name, value in metrics_dict.items():
                    results["PAS"]["acc"][stain_test][metric_name][site].append(value)

        print("Total:")
        for stain in stains:
            values = []
            for metric_name in metrics:
                val_list = results["PAS"]["acc"][stain][metric_name]["total"]
                values.append(val_list[-1] if val_list else float('nan'))
            print(f"\t(Test){stain} Best ACC model || " +
                  " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        for site in ["site1", "site2"]:
            print(f"\n{site.capitalize()}:")
            for stain in stains:
                values = []
                for metric_name in metrics:
                    val_list = results["PAS"]["acc"][stain][metric_name][site]
                    values.append(val_list[-1] if val_list else float('nan'))
                print(f"\t(Test){stain} Best ACC model || " +
                      " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        best_auc_model_weight_PAS = torch.load(os.path.join(args.fold_save_path, 'best_auc_PAS.pth'))
        model.load_state_dict(best_auc_model_weight_PAS)

        (
            acc_AFB, macro_auc_score_AFB, precision_AFB, recall_AFB, F1_score_AFB, specificity_AFB,
            sensitivity_AFB,
            acc_GMS, macro_auc_score_GMS, precision_GMS, recall_GMS, F1_score_GMS, specificity_GMS,
            sensitivity_GMS,
            acc_PAS, macro_auc_score_PAS, precision_PAS, recall_PAS, F1_score_PAS, specificity_PAS,
            sensitivity_PAS,
            site_metrics
        ) = eval(args, test_loader, model, eval_metric='_ON_PAS_BEST_AUC_MODEL')

        for stain_eval, values in overall_values.items():
            for metric_name, value in zip(metrics, values):
                results["PAS"]["auc"][stain_eval][metric_name]["total"].append(value)

        for site, site_data in site_metrics.items():  # site = "site1", "site2"
            for stain_test, metrics_dict in site_data.items():  # stain_test = "AFB", "GMS", "PAS"
                for metric_name, value in metrics_dict.items():
                    results["PAS"]["auc"][stain_test][metric_name][site].append(value)

        print("Total:")
        for stain in stains:
            values = []
            for metric_name in metrics:
                val_list = results["PAS"]["auc"][stain][metric_name]["total"]
                values.append(val_list[-1] if val_list else float('nan'))
            print(f"\t(Test){stain} Best AUC model || " +
                  " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

        for site in ["site1", "site2"]:
            print(f"\n{site.capitalize()}:")
            for stain in stains:
                values = []
                for metric_name in metrics:
                    val_list = results["PAS"]["auc"][stain][metric_name][site]
                    values.append(val_list[-1] if val_list else float('nan'))
                print(f"\t(Test){stain} Best AUC model || " +
                      " || ".join(f"{m.capitalize()}: {v:.6f}" for m, v in zip(metrics, values)))

    import numpy as np


    stains = ["AFB", "GMS", "PAS"]
    metrics = ["acc", "macro_auc", "precision", "recall", "F1", "specificity", "sensitivity"]
    sites = ["total", "site1", "site2"]

    print("Five-Fold-Validation Summary:")

    for m1, m2 in models:
        print(f"\nModel: {m1} - {m2}")
        for stain in stains:
            print(f"\tStain: {stain}")
            for site in sites:
                metric_strs = []
                for metric_name in metrics:
                    fold_values = results[m1][m2][stain][metric_name][site]
                    mean_val = np.mean(fold_values) * 100 if fold_values else float('nan')
                    std_val = np.std(fold_values) * 100 if fold_values else float('nan')
                    metric_strs.append(f"{metric_name.capitalize()}: {mean_val:.2f}±{std_val:.2f}")
                print(f"\t\t{site}: " + ", ".join(metric_strs))
