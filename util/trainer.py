from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.mixture import GaussianMixture
import os, logging, warnings
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')


def metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }
def matrix (y_true, y_pred):
    matri=classification_report(y_true, y_pred, target_names=['World', 'Sports', 'Business', 'Sci/Tech'], output_dict=True)
    return matri
       
def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


class SelfMixTrainer:
    def __init__(self, model, train_data=None, eval_data=None, model_args=None, training_args=None):
        self.model = model.cuda()
        self.train_data = train_data
        self.eval_data = eval_data
        self.model_args = model_args
        self.training_args = training_args
        if self.training_args is not None:
            self.optimizer = Adam(self.model.parameters(), lr=training_args.lr)
    
    def warmup(self):
        logger.info("***** Warmup stage *****")
        
        train_loader = self.train_data.run("all")
        if self.training_args.warmup_strategy == "epoch":
            warmup_samples = self.training_args.warmup_epochs * len(train_loader.dataset)
            warmup_epochs = self.training_args.warmup_epochs
        elif self.training_args.warmup_strategy == "samples":
            warmup_samples = self.training_args.warmup_samples
            warmup_epochs = self.training_args.warmup_samples // len(train_loader.dataset) + \
                            int(self.training_args.warmup_samples % len(train_loader.dataset) > 0)
        else:
            warmup_samples, warmup_epochs = 0, 0
            
        loss_func = nn.CrossEntropyLoss()
        now_samples = 0
        for epoch_id in range(1, warmup_epochs + 1):
            logger.info("***** Warmup epoch %d *****", epoch_id)
            
            self.model.train()
            train_loss, train_acc = 0., 0.
            for i, data in enumerate(train_loader):  
                input_ids, att_mask, labels, _  = [Variable(elem.cuda()) for elem in data]
                logits = self.model(input_ids, att_mask)
                loss = loss_func(logits, labels)
                train_loss += loss.item()
                
                pred = logits.argmax(dim=-1).cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                train_acc += (pred == labels).sum()

                loss = loss / self.training_args.grad_acc_steps
                loss.backward()

                if (i + 1) % self.training_args.grad_acc_steps == 0 or i + 1 == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                now_samples += input_ids.size(0)
                if now_samples >= warmup_samples:
                    logger.info("Warmup Stage ends in %d samples", now_samples)
                        
            logger.info("Warmup train samples [{:6d}/{:6d}], Loss: {:4f}, Accuracy: {:.2%}"
                        .format(now_samples, warmup_samples, train_loss / len(train_loader), train_acc / len(train_loader.dataset)))
                    
            if self.eval_data is not None:
                eval_loader = self.eval_data.run("all")
                self.evaluate(eval_loader)
                
    def evaluate(self, eval_loader=None):
        if eval_loader is None:
            eval_loader = self.eval_data.run("all")        
        self.model.eval()
        y_true, y_pred = np.zeros(len(eval_loader.dataset), dtype=int), np.zeros(len(eval_loader.dataset), dtype=int)
        for j, data in enumerate(eval_loader):
            val_input_ids, val_att, val_labels, index = [Variable(elem.cuda()) for elem in data]
            with torch.no_grad():
                index = index.long().cpu().detach().numpy()
                pred = self.model(val_input_ids, val_att).argmax(dim=-1).cpu().detach().numpy()
                val_labels = val_labels.cpu().detach().numpy()
            y_true[index] = val_labels
            y_pred[index] = pred
            
        eval_res = metric(y_true, y_pred)
      
        logger.info("Eval Results: Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, F1: {:.2%}"
            .format(eval_res['accuracy'], eval_res['precision'], eval_res['recall'], eval_res['f1']))
    def test(self, eval_loader=None):
        if eval_loader is None:
            eval_loader = self.eval_data.run("all")        
        self.model.eval()
        y_true, y_pred = np.zeros(len(eval_loader.dataset), dtype=int), np.zeros(len(eval_loader.dataset), dtype=int)
        for j, data in enumerate(eval_loader):
            val_input_ids, val_att, val_labels, index = [Variable(elem.cuda()) for elem in data]
            with torch.no_grad():
                index = index.long().cpu().detach().numpy()
                pred = self.model(val_input_ids, val_att).argmax(dim=-1).cpu().detach().numpy()
                val_labels = val_labels.cpu().detach().numpy()
            y_true[index] = val_labels
            y_pred[index] = pred
            
        eval_res = metric(y_true, y_pred)
        logger.info("Eval Results: Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, F1: {:.2%}"
            .format(eval_res['accuracy'], eval_res['precision'], eval_res['recall'], eval_res['f1']))
        evaluation_results = "Eval Results: Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, F1: {:.2%}".format(eval_res['accuracy'], eval_res['precision'], eval_res['recall'], eval_res['f1'])
        eval_matrix = matrix(y_true, y_pred)
        logger.info(eval_matrix)
        
        selecting_strategy=self.model_args.selection_strategy 
        noise_ratio = self.model_args.noised_rate
        noise_type = self.model_args.noise_type
        with open(f'./out_put/{selecting_strategy}.txt', 'a+') as f:
            
            f.write(f'\nEval Results: {noise_type}+{selecting_strategy}+{noise_ratio}')
            f.write("\n"+evaluation_results )
            f.write("\n"+str(eval_matrix))
        
            
            
            
    def train(self):
        logger.info("***** Mixup Train *****")       

        train_loader = self.train_data.run("all")
        eval_loader = self.eval_data.run("all")
        
        for epoch_id in range(1, self.training_args.train_epochs + 1):
            seletion_strategy = self.model_args.selection_strategy
            if seletion_strategy=='GMM':
                prob = self._eval_samples(train_loader)
                pred = (prob > self.model_args.p_threshold)
            else:
                prob = self._eval_samples_repesentive(train_loader,tamplate_tokenizer=self.model_args.token_representation)
                pred = (prob > 0)
                
            
            labeled_train_loader, unlabeled_train_loader = self.train_data.run("train", pred, prob)
            # return input_id, att_mask, self.labels[index], self.prob[index], self.pred_idx[index],self.inputs[index] # return of labeled
            inputs = []
            for i in range(len(labeled_train_loader.dataset)):
                probb,label,text =labeled_train_loader.dataset[i][3],labeled_train_loader.dataset[i][2],labeled_train_loader.dataset[i][5]
                
                # print(f'labeled :input {probb} label {label} text {text}')
                inputs.append(
                    {
                        "input":text,
                        "label":label,
                        "prob":probb,
                        "selection_strategy":seletion_strategy,
                        "noise_ratio":self.model_args.noised_rate,
                        "type":"labeled"
                    }
                )
                
            for i in range(len(unlabeled_train_loader.dataset)):
                probb,text,label =unlabeled_train_loader.dataset[i][2],unlabeled_train_loader.dataset[i][-2],unlabeled_train_loader.dataset[i][-1]
                print(f'unlabeled : input {prob[probb]} label {label} text {text}')   
                inputs.append(
                      
                    {  "input":text,
                        "label":label,
                        "prob":prob[probb],
                        "selection_strategy":seletion_strategy,
                        "noise_ratio":self.model_args.noised_rate,
                        "type":"unlabeled"   
                    }
                )  
            # csv writer 
            import csv
            with open(f'./output/{seletion_strategy}.csv', 'w+', newline='') as csvfile:
                fieldnames = ['input', 'label', 'prob','selection_strategy','noise_ratio','type']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for i in inputs:
                    writer.writerow(i)
            print('select_strategy',seletion_strategy)     
            print("Labeled: {}, Unlabeled: {}".format(len(labeled_train_loader.dataset), len(unlabeled_train_loader.dataset)))
            
           
                
            
            logger.info("***   Train epoch  %d ***", epoch_id)
            self._train_epoch(labeled_train_loader, unlabeled_train_loader)
            
            logger.info("*** Evaluate epoch %d ***", epoch_id)
            self.evaluate(eval_loader)
    
    def _train_epoch(self, labeled_train_loader, unlabeled_train_loader):
        labeled_train_iter = iter(labeled_train_loader)
        unlabeled_train_iter = iter(unlabeled_train_loader)
        val_iteration = len(labeled_train_loader)
        self.model.train()
        
        for batch_idx in range(val_iteration):
            try:
                inputs_x, inputs_x_att, targets_x, _,_,_ = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_train_loader)
                inputs_x, inputs_x_att, targets_x, _,_,_  = next(labeled_train_iter)

            try:
                inputs_u, att_u, _,_,_ = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_train_loader)
                inputs_u, att_u, _,_,_ = next(unlabeled_train_iter)

            
            targets_x = F.one_hot(targets_x, num_classes=self.model_args.num_classes)
            inputs_x, inputs_x_att, targets_x = inputs_x.cuda(), inputs_x_att.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, att_u = inputs_u.cuda(), att_u.cuda()
                        
            self.model.eval()
            with torch.no_grad():
                # Predict labels for unlabeled data.
                out_u = self.model(inputs_u, att_u)
                p = torch.softmax(out_u, dim=1)
                pt = p ** (1 / self.model_args.temp)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()
                
            self.model.train()
            sents_x = self.model.get_sentence_embedding(inputs_x, inputs_x_att)
            sents_u1 = self.model.get_sentence_embedding(inputs_u, att_u)
            sents_u2 = self.model.get_sentence_embedding(inputs_u, att_u)
            
            all_sents = torch.cat(
                [sents_x, sents_u1], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_u], dim=0)
            
            # mixup
            rand_idx = torch.randperm(all_sents.size(0))
            l = np.random.beta(self.model_args.alpha, self.model_args.alpha)
            l = max(l, 1 - l)
            mixed_sents = l * all_sents + (1 - l) * all_sents[rand_idx]
            mixed_targets = l * all_targets + (1 - l) * all_targets[rand_idx]
            
            logits = self.model.classify(mixed_sents)
            logits_u1 = self.model.classify(sents_u1)
            logits_u2 = self.model.classify(sents_u2)
            
            # compute loss
            loss_mix = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * mixed_targets, dim=-1))            
            pse_loss = -torch.mean(F.log_softmax(logits_u1, dim=1).min(dim=1)[0]) * 0.5 \
                        - torch.mean(F.log_softmax(logits_u2, dim=1).min(dim=1)[0]) * 0.5
            kl_loss = compute_kl_loss(logits_u1, logits_u2)
            loss = loss_mix + pse_loss * self.model_args.lambda_p + kl_loss * self.model_args.lambda_r
            loss = loss / self.training_args.grad_acc_steps
            loss.backward()
            if (batch_idx + 1) % self.training_args.grad_acc_steps == 0 or batch_idx + 1 == val_iteration:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
        logger.info("Loss_Mix: {:.4f}, Loss_P: {:.4f}, Loss_R: {:.4f}, Loss: {:.4f} "
                    .format(loss_mix, pse_loss, kl_loss, loss))

    def _eval_samples(self, eval_loader):
        """
        Sample selection
        """
        self.model.eval()
        
        loss_func = nn.CrossEntropyLoss(reduction='none')
        losses = np.zeros(len(eval_loader.dataset))
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                input_ids, att_mask, labels, index = [Variable(elem.cuda()) for elem in data] 
                outputs = self.model(input_ids, att_mask) 
                pred = torch.softmax(outputs, dim=-1)
                loss = loss_func(pred, labels).cpu().detach().numpy()
                index = index.long().cpu().detach().numpy()
                losses[index] = loss
                
        if self.model_args.class_reg:
            labels = np.array(eval_loader.dataset.labels, dtype=int)
            for now_class in range(self.model_args.num_classes):
                indices = np.where(labels == now_class)[0]
                losses[indices] = (losses[indices] - losses[indices].mean()) / losses[indices].var()
        else:
            losses = (losses - losses.min()) / (losses.max() - losses.min())
        
        gmm = GaussianMixture(
            n_components=2, 
            max_iter=self.model_args.gmm_max_iter, 
            tol=self.model_args.gmm_tol, 
            reg_covar=self.model_args.gmm_reg_covar
        )
        losses = losses.reshape(-1, 1)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses) 
        prob = prob[:,gmm.means_.argmin()]
        return prob
    def _eval_samples_repesentive(self, eval_loader,tamplate_tokenizer):
        """
        Sample selection
        """
        label_desc = [
        "It's a world news",
        "It's a sports news",
        "It's a business news",
        "It's a science news"
        ]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        label_encodings = tamplate_tokenizer(label_desc, return_tensors="pt").to(device)
        self.model.eval()
        
        # losses = np.zeros(len(eval_loader.dataset))
        # tmp_outputs = self.model(**label_encodings)
        label_outputs = self.model(label_encodings['input_ids'],label_encodings['attention_mask'])
        # Normalize the vectors
        # label_outputs = F.normalize(label_outputs, p=2, dim=1)
        
        losses = []   
        with torch.no_grad():
            for i, data in enumerate(eval_loader):        
                label_reps = label_outputs.mean(dim=1)  # Mean pooling
                input_ids, att_mask, labels, index = [Variable(elem.cuda()) for elem in data] 
                outputs = self.model(input_ids, att_mask) 
                # print('outputs ',outputs.shape)
                # outputs = F.normalize(outputs, p=2, dim=1)
                # news_embeddings =outputs.mean(dim=1)  # Mean pooling
                
                similarities = F.cosine_similarity(label_outputs.unsqueeze(0), outputs.unsqueeze(1), dim=2)
                # soft_max
                similarities = F.softmax(similarities/0.1, dim=1)
                
                loss = similarities.cpu().detach().numpy()
                true_labels = labels.cpu().detach().numpy()
                # loss  similarites big largest
                
                for i in range(len(loss)):
                    loss_i = loss[i]
                    loss_i = list(loss_i)
                    correct_label = true_labels[i]
                    correct_score = loss[i][correct_label]
                    loss_i.pop(correct_label)
                    loss_i = sorted(loss_i,reverse=True)
                    
                    losses.append(correct_score-max(loss_i))
        losses = np.array(losses)        
        return losses
    def _eval_samples_protype(self, eval_loader,tamplate_tokenizer):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for i, data in enumerate(eval_loader):  
                input_ids, att_mask, labels, index = [Variable(elem.cuda()) for elem in data] 
                outputs = self.model(input_ids, att_mask) 
                # consine similarity each output 
                # Normalize the vectors
                outputs
                
    def save_model(self):
        self.model.save_model(self.training_args.model_save_path)
