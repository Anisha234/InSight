import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,balanced_accuracy_score
from src import test
import numpy as np
import torch, time, gc
import torch.nn.functional as F

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))


def train(model, train_dataloader, val_dataloader,test_dataloader, criterion, optimizer, scheduler=None, num_epochs=50, backbone='Retina', save=False,save_dir=None, device='cpu', patience=7,use_amp=True):
    model.to(device)

    binary = True if train_dataloader.dataset.labels.shape[1] == 1 else False

    train_losses = []
    val_losses = []
    f1_scores = []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        'f1_score': 0.0,
        'ba': 0.0,    
    }
    epochs_no_improve = 0
    early_stop = False
    print("AMP", use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)
        start_timer()
        all_preds = []
        all_labels = []

        for num_batch,batch in enumerate(tqdm(train_dataloader, total=num_train_batches)):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                inputs = batch['image'] #.to(device)
                if inputs is not None:
                    if type(inputs) is list:
                        for idx in range(len(inputs)):
                            inputs[idx] = inputs[idx].to(device)
                    else:
                        inputs = inputs.to(device)
                labels = batch['labels'].to(device)
                
                if 'meta' in batch:
                    meta = batch['meta'].to(device)
                    outputs = model(inputs, meta)
                else:
                    outputs = model(inputs)


                if binary:
                    loss = criterion(outputs, labels.float())
                else:
                    loss = criterion(outputs, torch.argmax(labels, dim=1))
                preds = torch.argmax(outputs, dim=1) if not binary else outputs.round()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(torch.argmax(labels, dim=1).cpu().numpy() if not binary else labels.cpu().numpy())
         #   loss.backward()
        #    optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance
            total_loss += loss.item()
            #if num_batch == 100:
            #    end_timer_and_print("Test")
                
       # end_timer_and_print("Test")

        avg_train_loss = total_loss / num_train_batches
        train_losses.append(avg_train_loss)
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        ba = balanced_accuracy_score(all_labels, all_preds)

        f1_scores.append(f1)
        confusion_matrix_sc = confusion_matrix(all_labels, all_preds)
        print(f'Epoch {epoch + 1}')
        print("Train loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_train_loss, c=f1, d = acc, e= ba))
        print(f"cm{confusion_matrix_sc}")
        
        if scheduler is not None:
            print("updating LR")
            scheduler.step()
            
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, total=len(val_dataloader)):
                val_inputs = val_batch['image']
                val_labels = val_batch['labels'].to(device)
                if val_inputs is not None:
                    if type(val_inputs) is list:
                        for idx in range(len(val_inputs)):
                            val_inputs[idx] = val_inputs[idx].to(device)
                    else:
                        val_inputs = val_inputs.to(device)
                if 'meta' in val_batch:
                    meta = val_batch['meta'].to(device)
                    val_outputs = model(val_inputs, meta)
                else:
                    val_outputs = model(val_inputs)

                if binary:
                    val_loss += criterion(val_outputs, val_labels.float()).item()
                else:
                    val_loss += criterion(val_outputs, torch.argmax(val_labels, dim=1)).item()

                preds = torch.argmax(val_outputs, dim=1) if not binary else val_outputs.round()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(torch.argmax(val_labels, dim=1).cpu().numpy() if not binary else val_labels.cpu().numpy())

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        ba = balanced_accuracy_score(all_labels, all_preds)
        f1_scores.append(f1)
        confusion_matrix_sc = confusion_matrix(all_labels, all_preds)
        print(f'Epoch {epoch + 1}')
        print("Val loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=val_loss, c=f1, d = acc, e= ba))
        print(f"cm{confusion_matrix_sc}")

        test.test(model, test_dataloader, saliency=False, device='cuda', save=False)
        if f1 > best_model_info['f1_score']:
            best_model_info['epoch'] = epoch + 1
            best_model_info['state_dict'] = model.state_dict()
            best_model_info['f1_score'] = f1
            best_model_info['ba'] = ba
            epochs_no_improve = 0
            if save:
                print("Saving")
                os.makedirs(save_dir, exist_ok=True)
                model.load_state_dict(best_model_info['state_dict'])
                torch.save(model.state_dict(), save_dir+'\\fine_tuned_resnet50_best.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            early_stop = True
            
            break

    if not early_stop:
        print('Training completed without early stopping.')
        
    # Load best model
    if best_model_info['state_dict'] is not None:
        model.load_state_dict(best_model_info['state_dict'])

  
    return model

## Code to edit
def multitask_train(model,task_index, train_dataloader, val_dataloader,test_dataloader, criterion_list, optimizer, scheduler=None, num_epochs=100, backbone='Retina', save=False,save_dir=None, device='cpu', patience=7,use_amp=True):
    model.to(device)

    #binary = True if train_dataloader.dataset.labels.shape[1] == 1 else False

    train_losses = []
    val_losses = []
    f1_scores = [[],[],[],[],[],[],[],[],[]]
    val_f1_scores= []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        'f1_score': 0.0,  #pick best model based on highest f1 score for dr classification task
        'ba': 0.0,    
    }
    epochs_no_improve = 0
    early_stop = False
    print("AMP", use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_task_loss= [0,0,0,0,0,0,0,0,0]
        avg_train_loss = total_task_loss
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)
        start_timer()
        all_preds = [[],[],[],[],[],[],[],[],[]]
        all_labels = [[],[],[],[],[],[],[],[],[]]

        for num_batch,batch in enumerate(tqdm(train_dataloader, total=num_train_batches)):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                inputs = batch['image'] #.to(device)
                for idx in range(len(inputs)):
                    inputs[idx] = inputs[idx].to(device)
                labels = batch['labels'].to(device)
                meta = batch['meta'].to(device)
                outputs = model(inputs, meta)
                #teacher_soft_labels = F.softmax(outputs[0], dim=1)  # Corrected
                #print(teacher_soft_labels, labels[:,0])
                loss = 0
                loss_task = torch.zeros(len(criterion_list))
                for i in range(len(criterion_list)):
                    if i in task_index:
                        loss_task[i] = criterion_list[i](outputs[i], labels[:,i])
                        loss += loss_task[i]
                    preds = torch.argmax(outputs[i], dim=1) 
                    all_preds[i].extend(preds.cpu().numpy()) #across an epoch
                    all_labels[i].extend(labels[:,i].cpu().numpy()) #across an epoch
         #   loss.backward()
        #    optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance
            total_loss += loss.item()
            for i in range(len(criterion_list)):
                total_task_loss[i] += loss_task[i].item()
            #if num_batch == 100:
            #    end_timer_and_print("Test")
                
       # end_timer_and_print("Test")

       # avg_train_loss = total_loss / num_train_batches
        for i in range(len(criterion_list)):
            avg_train_loss[i] = total_task_loss[i]/num_train_batches

        train_losses.append(avg_train_loss)
        for i in range(min(len(criterion_list),6)):
            f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
            f1_scores[i].append(f1) #f1scores stores across epochs
            acc = accuracy_score(all_labels[i], all_preds[i])
            ba = balanced_accuracy_score(all_labels[i], all_preds[i])
            confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
            print(f'Epoch {epoch + 1}')
            print("Train loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_train_loss[i], c=f1, d = acc, e= ba))
            print(f"cm{confusion_matrix_sc}")
        
        if scheduler is not None:
            scheduler.step()
        if 1:
            model.eval()
            avg_val_loss = 0.0
            all_preds = [[],[],[],[],[],[],[],[],[]]
            all_labels = [[],[],[],[],[],[],[],[],[]]
            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, total=len(val_dataloader)):
                    val_inputs = val_batch['image']
                    val_labels = val_batch['labels'].to(device)
                    for idx in range(len(val_inputs)):
                        val_inputs[idx] = val_inputs[idx].to(device)
                    meta = val_batch['meta'].to(device)
                    val_outputs = model(val_inputs, meta)
                    for i in range(min(len(criterion_list),6)): #change to 6 !!!
                        if i in task_index:
                            loss = criterion_list[i](val_outputs[i], val_labels[:,i])
                            avg_val_loss += loss.item()
                        preds = torch.argmax(val_outputs[i], dim=1) 
                        all_preds[i].extend(preds.cpu().numpy())
                        all_labels[i].extend(val_labels[:,i].cpu().numpy())
                   
    
                    
    
            avg_val_loss /= len(val_dataloader)
            val_losses.append(avg_val_loss)
            val_f1_scores =[]
            val_ba_scores =[]
            for i in range(min(len(criterion_list),6)):
                f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
                acc = accuracy_score(all_labels[i], all_preds[i])
                ba = balanced_accuracy_score(all_labels[i], all_preds[i])
                if i in task_index:
                    val_f1_scores.append(f1) #f1scores stores across epochs
                    val_ba_scores.append(ba)
                if(i == 0):
                    ba_score_icdr = ba
                confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
                print(f'Epoch {epoch + 1}')
                print("Val loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_val_loss, c=f1, d = acc, e= ba))
                print(f"cm{confusion_matrix_sc}")
            # Mean f1 score across all the tasks
            f1_score_mean = np.mean(val_f1_scores)
            ba_score_mean = np.mean(val_ba_scores)
    #        f1_score_mean = np.mean(val_f1_scores[:6])
            print("Mean f1 score {a:.3f}".format(a= f1_score_mean))
            print("Mean ba score {a:.3f}".format(a= ba_score_mean))
        test.test_multitask(model,task_index, test_dataloader, criterion_list, saliency=False, device=device)
        if ba_score_mean > best_model_info['ba']:
            best_model_info['epoch'] = epoch + 1
            best_model_info['state_dict'] = model.state_dict()
            best_model_info['f1_score'] = f1_score_mean
            best_model_info['ba'] = ba_score_mean
            epochs_no_improve = 0
            if save:
                print("Saving")
                os.makedirs(save_dir, exist_ok=True)
                model.load_state_dict(best_model_info['state_dict'])
                torch.save(model.state_dict(), save_dir+'\\fine_tuned_model_best.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            early_stop = True
            
            break

    if not early_stop:
        print('Training completed without early stopping.')
        
    # Load best model
    if best_model_info['state_dict'] is not None:
        model.load_state_dict(best_model_info['state_dict'])

  
    return model
## Code to edit

def multitask_joint_train(model,task_index, train_dataloader, val_dataloader,test_dataloader, criterion_list1,criterion_list2, optimizer, scheduler=None, num_epochs=100, backbone='Retina', save=False,save_dir=None, device='cpu', patience=7,use_amp=True):
    model.to(device)

    #binary = True if train_dataloader.dataset.labels.shape[1] == 1 else False

    train_losses = []
    val_losses = []
    f1_scores = [[],[],[],[],[],[],[],[],[]]
    val_f1_scores= []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        'f1_score': 0.0,  #pick best model based on highest f1 score for dr classification task
        'ba': 0.0,    
    }
    epochs_no_improve = 0
    early_stop = False
    print("AMP", use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_task_loss= [0,0,0,0,0,0,0,0,0]
        avg_train_loss = total_task_loss
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)
        start_timer()
        all_preds = [[],[],[],[],[],[],[],[],[]]
        all_labels = [[],[],[],[],[],[],[],[],[]]

        for num_batch,batch in enumerate(tqdm(train_dataloader, total=num_train_batches)):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                inputs = batch['image'] #.to(device)
                for idx in range(len(inputs)):
                    inputs[idx] = inputs[idx].to(device)
                labelsFull = batch['labels'].to(device)
                labels1 = labelsFull[:,:4]
                labels2 = labelsFull[:,4:]
                meta = batch['meta'].to(device)
                outputs1,outputs2 = model(inputs, meta)
                #teacher_soft_labels = F.softmax(outputs[0], dim=1)  # Corrected
                #print(teacher_soft_labels, labels[:,0])
                loss = 0
                loss_task = torch.zeros(len(criterion_list1))
                for i in range(len(criterion_list1)):
                    if i in task_index:
                        loss_task[i] = criterion_list1[i](outputs1[i], labels1[:,i])
                        #loss_task[i] = criterion_list2[i](outputs2[i], labels2[:,i])
                        loss += loss_task[i]
                    preds1 = torch.argmax(outputs1[i], dim=1) 
                    all_preds[i].extend(preds1.cpu().numpy()) #across an epoch
                    all_labels[i].extend(labels1[:,i].cpu().numpy()) #across an epoch
                    preds2 = torch.argmax(outputs2[i], dim=1) 
                    all_preds[i].extend(preds2.cpu().numpy()) #across an epoch
                    all_labels[i].extend(labels2[:,i].cpu().numpy()) #across an epoch
         #   loss.backward()
        #    optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance
            total_loss += loss.item()
            for i in range(len(criterion_list1)):
                total_task_loss[i] += loss_task[i].item()
            #if num_batch == 100:
            #    end_timer_and_print("Test")
                
       # end_timer_and_print("Test")

       # avg_train_loss = total_loss / num_train_batches
        for i in range(len(criterion_list1)):
            avg_train_loss[i] = total_task_loss[i]/num_train_batches

        train_losses.append(avg_train_loss)
        for i in range(min(len(criterion_list1),6)):
            f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
            f1_scores[i].append(f1) #f1scores stores across epochs
            acc = accuracy_score(all_labels[i], all_preds[i])
            ba = balanced_accuracy_score(all_labels[i], all_preds[i])
            confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
            print(f'Epoch {epoch + 1}')
            print("Train loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_train_loss[i], c=f1, d = acc, e= ba))
            print(f"cm{confusion_matrix_sc}")
        
        if scheduler is not None:
            scheduler.step()
            
        model.eval()
        avg_val_loss = 0.0
        all_preds = [[],[],[],[],[],[],[],[],[]]
        all_labels = [[],[],[],[],[],[],[],[],[]]
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, total=len(val_dataloader)):
                val_inputs = val_batch['image']
                val_labels = val_batch['labels'].to(device)
                val_labels_full = val_batch['labels'].to(device)
                val_labels1 = val_labels_full[:,:4]
                val_labels2 = val_labels_full[:,4:]
                for idx in range(len(val_inputs)):
                    val_inputs[idx] = val_inputs[idx].to(device)
                meta = val_batch['meta'].to(device)
                val_outputs1,val_outputs2 = model(val_inputs, meta)

                for i in range(min(len(criterion_list1),6)): #change to 6 !!!
                    if i in task_index:
                        loss = criterion_list1[i](val_outputs1[i], val_labels1[:,i])
                        avg_val_loss += loss.item()
                        loss = criterion_list2[i](val_outputs2[i], val_labels2[:,i])
                        avg_val_loss += loss.item()
                    preds1 = torch.argmax(val_outputs1[i], dim=1) 
                    all_preds[i].extend(preds1.cpu().numpy())
                    all_labels[i].extend(val_labels1[:,i].cpu().numpy())
                    preds2 = torch.argmax(val_outputs2[i], dim=1) 
                    all_preds[i].extend(preds2.cpu().numpy())
                    all_labels[i].extend(val_labels2[:,i].cpu().numpy())
               

                

        avg_val_loss /= len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_f1_scores =[]
        val_ba_scores =[]
        for i in range(min(len(criterion_list1),6)):
            f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
            acc = accuracy_score(all_labels[i], all_preds[i])
            ba = balanced_accuracy_score(all_labels[i], all_preds[i])
            if i in task_index:
                val_f1_scores.append(f1) #f1scores stores across epochs
                val_ba_scores.append(ba)
            if(i == 0):
                ba_score_icdr = ba
            confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
            print(f'Epoch {epoch + 1}')
            print("Val loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_val_loss, c=f1, d = acc, e= ba))
            print(f"cm{confusion_matrix_sc}")
        # Mean f1 score across all the tasks
        f1_score_mean = np.mean(val_f1_scores)
        ba_score_mean = np.mean(val_ba_scores)
#        f1_score_mean = np.mean(val_f1_scores[:6])
        print("Mean f1 score {a:.3f}".format(a= f1_score_mean))
        print("Mean ba score {a:.3f}".format(a= ba_score_mean))
        test.test_joint_multitask(model,task_index, test_dataloader, criterion_list1,criterion_list2, saliency=False, device=device)
        if ba_score_mean > best_model_info['ba']:
            best_model_info['epoch'] = epoch + 1
            best_model_info['state_dict'] = model.state_dict()
            best_model_info['f1_score'] = f1_score_mean
            best_model_info['ba'] = ba_score_mean
            epochs_no_improve = 0
            if save:
                print("Saving")
                os.makedirs(save_dir, exist_ok=True)
                model.load_state_dict(best_model_info['state_dict'])
                torch.save(model.state_dict(), save_dir+'\\fine_tuned_resnet50_best.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            early_stop = True
            
            break

    if not early_stop:
        print('Training completed without early stopping.')
        
    # Load best model
    if best_model_info['state_dict'] is not None:
        model.load_state_dict(best_model_info['state_dict'])

  
    return model
## Code to edit
def multitask_pretrain(model,task_index, train_dataloader, val_dataloader,test_dataloader, criterion_list, criterion_img,optimizer, scheduler=None, num_epochs=100, backbone='Retina', save=False,save_dir=None, device='cpu', patience=7,use_amp=True):
    model.to(device)

    #binary = True if train_dataloader.dataset.labels.shape[1] == 1 else False

    train_losses = []
    val_losses = []
    f1_scores = [[],[],[],[],[],[],[],[],[]]
    val_f1_scores= []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        'f1_score': 0.0,  #pick best model based on highest f1 score for dr classification task
        'ba': 0.0,    
    }
    epochs_no_improve = 0
    early_stop = False
    print("AMP", use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_task_loss= [0,0,0,0,0,0,0,0,0]
        total_img_loss = 0
        avg_train_loss = total_task_loss
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)
        start_timer()
        all_preds = [[],[],[],[],[],[],[],[],[]]
        all_labels = [[],[],[],[],[],[],[],[],[]]

        for num_batch,batch in enumerate(tqdm(train_dataloader, total=num_train_batches)):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                inputs = batch['image'] #.to(device)
                for idx in range(len(inputs)):
                    inputs[idx] = inputs[idx].to(device)
                labels = batch['labels'].to(device)
                meta = batch['meta'].to(device)
                outputs,rec_img = model(inputs, meta)
                #teacher_soft_labels = F.softmax(outputs[0], dim=1)  # Corrected
                #print(teacher_soft_labels, labels[:,0])
                loss = 0
                loss_task = torch.zeros(len(criterion_list))
                loss_img = criterion_img(rec_img,inputs[0])
                for i in range(len(criterion_list)):
                    if i in task_index:
                        if i ==0:
                            wt = 0.1
                        else:
                            wt = 1
                        loss_task[i] = wt*criterion_list[i](outputs[i], labels[:,i])
                        loss += loss_task[i]
                    preds = torch.argmax(outputs[i], dim=1) 
                    all_preds[i].extend(preds.cpu().numpy()) #across an epoch
                    all_labels[i].extend(labels[:,i].cpu().numpy()) #across an epoch
         #   loss.backward()
        #    optimizer.step()
            scaler.scale(loss+loss_img).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance
            total_loss += loss.item()
            for i in range(len(criterion_list)):
                total_task_loss[i] += loss_task[i].item()
            total_img_loss +=loss_img.item()
            #if num_batch == 100:
            #    end_timer_and_print("Test")
                
       # end_timer_and_print("Test")

       # avg_train_loss = total_loss / num_train_batches
        for i in range(len(criterion_list)):
            avg_train_loss[i] = total_task_loss[i]/num_train_batches
        avg_img_loss= total_img_loss/num_train_batches
        train_losses.append(avg_train_loss)
        for i in range(min(len(criterion_list),6)):
            f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
            f1_scores[i].append(f1) #f1scores stores across epochs
            acc = accuracy_score(all_labels[i], all_preds[i])
            ba = balanced_accuracy_score(all_labels[i], all_preds[i])
            confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
            print(f'Epoch {epoch + 1}')
            print("Train loss {b:.3f}, Img loss {f:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_train_loss[i], f=avg_img_loss,c=f1, d = acc, e= ba))
            print(f"cm{confusion_matrix_sc}")
        
        if scheduler is not None:
            scheduler.step()
            
        model.eval()
        avg_val_loss = 0.0
        all_preds = [[],[],[],[],[],[],[],[],[]]
        all_labels = [[],[],[],[],[],[],[],[],[]]
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, total=len(val_dataloader)):
                val_inputs = val_batch['image']
                val_labels = val_batch['labels'].to(device)
                for idx in range(len(val_inputs)):
                    val_inputs[idx] = val_inputs[idx].to(device)
                meta = val_batch['meta'].to(device)
                val_outputs,rec_img= model(val_inputs, meta)
                for i in range(min(len(criterion_list),6)): #change to 6 !!!
                    if i in task_index:
                        loss = criterion_list[i](val_outputs[i], val_labels[:,i])
                        avg_val_loss += loss.item()
                    preds = torch.argmax(val_outputs[i], dim=1) 
                    all_preds[i].extend(preds.cpu().numpy())
                    all_labels[i].extend(val_labels[:,i].cpu().numpy())
               

                

        avg_val_loss /= len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_f1_scores =[]
        val_ba_scores =[]
        for i in range(min(len(criterion_list),6)):
            f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
            acc = accuracy_score(all_labels[i], all_preds[i])
            ba = balanced_accuracy_score(all_labels[i], all_preds[i])
            if i in task_index:
                val_f1_scores.append(f1) #f1scores stores across epochs
                val_ba_scores.append(ba)
            if(i == 0):
                ba_score_icdr = ba
            confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
            print(f'Epoch {epoch + 1}')
            print("Val loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_val_loss, c=f1, d = acc, e= ba))
            print(f"cm{confusion_matrix_sc}")
        # Mean f1 score across all the tasks
        f1_score_mean = np.mean(val_f1_scores)
        ba_score_mean = np.mean(val_ba_scores)
#        f1_score_mean = np.mean(val_f1_scores[:6])
        print("Mean f1 score {a:.3f}".format(a= f1_score_mean))
        print("Mean ba score {a:.3f}".format(a= ba_score_mean))
#        test.test_multitask(model,task_index, test_dataloader, criterion_list, saliency=False, device=device)
        if ba_score_mean > best_model_info['ba']:
            best_model_info['epoch'] = epoch + 1
            best_model_info['state_dict'] = model.state_dict()
            best_model_info['f1_score'] = f1_score_mean
            best_model_info['ba'] = ba_score_mean
            epochs_no_improve = 0
            if save:
                print("Saving")
                os.makedirs(save_dir, exist_ok=True)
                model.load_state_dict(best_model_info['state_dict'])
                torch.save(model.state_dict(), save_dir+'\\fine_tuned_resnet50_best.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            early_stop = True
            
            break

    if not early_stop:
        print('Training completed without early stopping.')
        
    # Load best model
    if best_model_info['state_dict'] is not None:
        model.load_state_dict(best_model_info['state_dict'])

  
    return model
def train_distill(model,modelT1, modelT2, task_index, train_dataloader, val_dataloader,test_dataloader, criterion_list,criterion_soft, temperature, optimizer, scheduler=None, num_epochs=100, backbone='Retina', save=False,save_dir=None, device='cpu', patience=7,use_amp=True):
    model.to(device)
    modelT1.to(device)
    modelT2.to(device)
    alpha =0.2
    #binary = True if train_dataloader.dataset.labels.shape[1] == 1 else False

    train_losses = []
    val_losses = []
    f1_scores = [[],[],[],[],[],[],[],[],[]]
    val_f1_scores= []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        'f1_score': 0.0,  #pick best model based on highest f1 score for dr classification task
        'ba': 0.0,    
    }
    epochs_no_improve = 0
    early_stop = False
    print("AMP", use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    modelT1.eval()
    modelT2.eval()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_task_loss= [0,0,0,0,0,0,0,0,0]
        avg_train_loss = total_task_loss
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)
        all_preds = [[],[],[],[],[],[],[],[],[]]
        all_labels = [[],[],[],[],[],[],[],[],[]]

        for num_batch,batch in enumerate(tqdm(train_dataloader, total=num_train_batches)):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                inputs = batch['image'] #.to(device)
                for idx in range(len(inputs)):
                    inputs[idx] = inputs[idx].to(device)
                labels = batch['labels'].to(device)
                meta = batch['meta'].to(device)
                outputs = model(inputs, meta)
                with torch.no_grad():
                    outputT1 = modelT1(inputs,meta)
                teacher_soft_labels = F.softmax(outputT1[0] / temperature, dim=1)  # Corrected
                student_soft_labels = F.log_softmax(outputs[0] / temperature, dim=1)
                # Compute Distillation Loss (Soft) + Cross Entropy Loss (Hard)
                loss_softT1 = criterion_soft(student_soft_labels, teacher_soft_labels)#* (temperature ** 2)
                #print(teacher_soft_labels, labels[:,0])
                with torch.no_grad():
                    outputT2 = modelT2(inputs,meta)
                teacher_soft_labels = F.softmax(outputT2[0] / temperature, dim=1)
                student_soft_labels = F.log_softmax(outputs[5] / temperature, dim=1)
                # Compute Distillation Loss (Soft) + Cross Entropy Loss (Hard)
                loss_softT2 = criterion_soft(student_soft_labels, teacher_soft_labels)#* (temperature ** 2)
               # print(teacher_soft_labels, labels[:,5])

   
                
                loss = 0
                loss_task = torch.zeros(len(criterion_list))
                for i in range(len(criterion_list)):
                    if i in task_index:
                        loss_task[i] = criterion_list[i](outputs[i], labels[:,i])
                        if i==0:
                 #           print(i, loss_softT1, loss_task[i])
                            loss_task[i] = alpha * loss_softT1 + (1 - alpha) * loss_task[i]
                        if i==5:
                  #          print(i, loss_softT2, loss_task[i])
                            loss_task[i] = alpha * loss_softT2 + (1 - alpha) * loss_task[i]
                            
                        loss += loss_task[i]
                    preds = torch.argmax(outputs[i], dim=1) 
                    all_preds[i].extend(preds.cpu().numpy()) #across an epoch
                    all_labels[i].extend(labels[:,i].cpu().numpy()) #across an epoch
         #   loss.backward()
        #    optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance
            total_loss += loss.item()
            for i in range(len(criterion_list)):
                total_task_loss[i] += loss_task[i].item()
            #if num_batch == 100:
            #    end_timer_and_print("Test")
                
       # end_timer_and_print("Test")

       # avg_train_loss = total_loss / num_train_batches
        for i in range(len(criterion_list)):
            avg_train_loss[i] = total_task_loss[i]/num_train_batches

        train_losses.append(avg_train_loss)
        for i in range(len(criterion_list)):
            f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
            f1_scores[i].append(f1) #f1scores stores across epochs
            acc = accuracy_score(all_labels[i], all_preds[i])
            ba = balanced_accuracy_score(all_labels[i], all_preds[i])
            confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
            print(f'Epoch {epoch + 1}')
            print("Train loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_train_loss[i], c=f1, d = acc, e= ba))
            print(f"cm{confusion_matrix_sc}")
        
        if scheduler is not None:
            scheduler.step()

        model.eval()
        avg_val_loss = 0.0
        all_preds = [[],[],[],[],[],[],[],[],[]]
        all_labels = [[],[],[],[],[],[],[],[],[]]
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, total=len(val_dataloader)):
                val_inputs = val_batch['image']
                val_labels = val_batch['labels'].to(device)
                for idx in range(len(val_inputs)):
                    val_inputs[idx] = val_inputs[idx].to(device)
                meta = val_batch['meta'].to(device)
                val_outputs = model(val_inputs, meta)
                for i in range(len(criterion_list)): #change to 6 !!!
                    if i in task_index:
                        loss = criterion_list[i](val_outputs[i], val_labels[:,i])
                        avg_val_loss += loss.item()
                    preds = torch.argmax(val_outputs[i], dim=1) 
                    all_preds[i].extend(preds.cpu().numpy())
                    all_labels[i].extend(val_labels[:,i].cpu().numpy())
               

                

        avg_val_loss /= len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_f1_scores =[]
        val_ba_scores =[]
        for i in range(len(criterion_list)):
            f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
            acc = accuracy_score(all_labels[i], all_preds[i])
            ba = balanced_accuracy_score(all_labels[i], all_preds[i])
            if i in task_index:
                val_f1_scores.append(f1) #f1scores stores across epochs
                val_ba_scores.append(ba)
            if(i == 0):
                ba_score_icdr = ba
            confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
            print(f'Epoch {epoch + 1}')
            print("Val loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_val_loss, c=f1, d = acc, e= ba))
            print(f"cm{confusion_matrix_sc}")
        # Mean f1 score across all the tasks
        f1_score_mean = np.mean(val_f1_scores)
        ba_score_mean = np.mean(val_ba_scores)
#        f1_score_mean = np.mean(val_f1_scores[:6])
        print("Mean f1 score {a:.3f}".format(a= f1_score_mean))
        print("Mean ba score {a:.3f}".format(a= ba_score_mean))
        test.test_multitask(model,task_index, test_dataloader, criterion_list, saliency=False, device=device)
        if ba_score_mean > best_model_info['ba']:
            best_model_info['epoch'] = epoch + 1
            best_model_info['state_dict'] = model.state_dict()
            best_model_info['f1_score'] = f1_score_mean
            best_model_info['ba'] = ba_score_mean
            epochs_no_improve = 0
            if save:
                print("Saving")
                os.makedirs(save_dir, exist_ok=True)
                model.load_state_dict(best_model_info['state_dict'])
                torch.save(model.state_dict(), save_dir+'\\fine_tuned_resnet50_best.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            early_stop = True
            
            break

    if not early_stop:
        print('Training completed without early stopping.')
        
    # Load best model
    if best_model_info['state_dict'] is not None:
        model.load_state_dict(best_model_info['state_dict'])

  
    return model

def train_DR(model,task_index, train_dataloader, val_dataloader,test_dataloader, criterion_list, optimizer, scheduler=None, num_epochs=100, backbone='Retina', save=False,save_dir=None, device='cpu', patience=7,use_amp=True):
    model.to(device)

    #binary = True if train_dataloader.dataset.labels.shape[1] == 1 else False

    train_losses = []
    val_losses = []
    f1_scores = [[],[],[],[],[],[],[],[],[]]
    val_f1_scores= []

    best_model_info = {
        'epoch': 0,
        'state_dict': None,
        'f1_score': 0.0,  #pick best model based on highest f1 score for dr classification task
        'ba': 0.0,    
    }
    epochs_no_improve = 0
    early_stop = False
    print("AMP", use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_task_loss= [0,0,0,0,0,0,0,0,0]
        avg_train_loss = total_task_loss
        total_accuracy = 0.0
        num_train_batches = len(train_dataloader)
        start_timer()
        all_preds = [[],[],[],[],[],[],[],[],[]]
        all_labels = [[],[],[],[],[],[],[],[],[]]

        for num_batch,batch in enumerate(tqdm(train_dataloader, total=num_train_batches)):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                inputs = batch['image'] #.to(device)
                for idx in range(len(inputs)):
                    inputs[idx] = inputs[idx].to(device)
                labels = batch['labels'].to(device)
                meta = batch['meta'].to(device)
                outputs = model(inputs, meta)
                #teacher_soft_labels = F.softmax(outputs[0], dim=1)  # Corrected
                #print(teacher_soft_labels, labels[:,0])
                loss = 0
                loss_task = torch.zeros(len(criterion_list))
                for i in range(len(criterion_list)):
                    if i in task_index:
                        loss_task[i] = criterion_list[i](outputs[i], labels[:,i].float())
                        loss += loss_task[i]
                    #preds = torch.argmax(outputs[i], dim=1) 
                    
                    preds = torch.round(outputs[i]).long()
                    all_preds[i].extend(preds.cpu().numpy()) #across an epoch
                    all_labels[i].extend(labels[:,i].cpu().numpy()) #across an epoch
         #   loss.backward()
        #    optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance
            total_loss += loss.item()
            for i in range(len(criterion_list)):
                total_task_loss[i] += loss_task[i].item()
            #if num_batch == 100:
            #    end_timer_and_print("Test")
                
       # end_timer_and_print("Test")

       # avg_train_loss = total_loss / num_train_batches
        for i in range(len(criterion_list)):
            avg_train_loss[i] = total_task_loss[i]/num_train_batches

        train_losses.append(avg_train_loss)
        for i in range(min(len(criterion_list),6)):
            f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
            f1_scores[i].append(f1) #f1scores stores across epochs
            acc = accuracy_score(all_labels[i], all_preds[i])
            ba = balanced_accuracy_score(all_labels[i], all_preds[i])
            confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
            print(f'Epoch {epoch + 1}')
            print("Train loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_train_loss[i], c=f1, d = acc, e= ba))
            print(f"cm{confusion_matrix_sc}")
        
        if scheduler is not None:
            scheduler.step()
        if 1:
            model.eval()
            avg_val_loss = 0.0
            all_preds = [[],[],[],[],[],[],[],[],[]]
            all_labels = [[],[],[],[],[],[],[],[],[]]
            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, total=len(val_dataloader)):
                    val_inputs = val_batch['image']
                    val_labels = val_batch['labels'].to(device)
                    for idx in range(len(val_inputs)):
                        val_inputs[idx] = val_inputs[idx].to(device)
                    meta = val_batch['meta'].to(device)
                    val_outputs = model(val_inputs, meta)
                    for i in range(min(len(criterion_list),6)): #change to 6 !!!
                        if i in task_index:
                            loss = criterion_list[i](val_outputs[i], val_labels[:,i].float())
                            avg_val_loss += loss.item()
                        preds = torch.round(val_outputs[i]).long()
                        all_preds[i].extend(preds.cpu().numpy())
                        all_labels[i].extend(val_labels[:,i].cpu().numpy())
                   
    
                    
    
            avg_val_loss /= len(val_dataloader)
            val_losses.append(avg_val_loss)
            val_f1_scores =[]
            val_ba_scores =[]
            for i in range(min(len(criterion_list),6)):
                f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
                acc = accuracy_score(all_labels[i], all_preds[i])
                ba = balanced_accuracy_score(all_labels[i], all_preds[i])
                if i in task_index:
                    val_f1_scores.append(f1) #f1scores stores across epochs
                    val_ba_scores.append(ba)
                if(i == 0):
                    ba_score_icdr = ba
                confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
                print(f'Epoch {epoch + 1}')
                print("Val loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_val_loss, c=f1, d = acc, e= ba))
                print(f"cm{confusion_matrix_sc}")
            # Mean f1 score across all the tasks
            f1_score_mean = np.mean(val_f1_scores)
            ba_score_mean = np.mean(val_ba_scores)
    #        f1_score_mean = np.mean(val_f1_scores[:6])
            print("Mean f1 score {a:.3f}".format(a= f1_score_mean))
            print("Mean ba score {a:.3f}".format(a= ba_score_mean))
        test.test_DR(model,task_index, test_dataloader, criterion_list, saliency=False, device=device)
        if ba_score_mean > best_model_info['ba']:
            best_model_info['epoch'] = epoch + 1
            best_model_info['state_dict'] = model.state_dict()
            best_model_info['f1_score'] = f1_score_mean
            best_model_info['ba'] = ba_score_mean
            epochs_no_improve = 0
            if save:
                print("Saving")
                os.makedirs(save_dir, exist_ok=True)
                model.load_state_dict(best_model_info['state_dict'])
                torch.save(model.state_dict(), save_dir+'\\fine_tuned_resnet50_best.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            early_stop = True
            
            break

    if not early_stop:
        print('Training completed without early stopping.')
        
    # Load best model
    if best_model_info['state_dict'] is not None:
        model.load_state_dict(best_model_info['state_dict'])

  
    return model
## Code to edit