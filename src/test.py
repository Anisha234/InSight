from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch 
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

# Metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay

# Generate Saliency Maps
def get_saliency_map(model, input_image):
    model.eval()
    input_image.requires_grad_()
    output = model(input_image)
    max_idx = output.argmax()
    output[0, max_idx].backward()
    saliency_map, _ = torch.max(input_image.grad.data.abs(),dim=1)
    #saliency_map = input_image.grad.data.abs().max(1)[0]
    return saliency_map

def test_model(y_test, y_pred, y_prob=None):
    """
    Evaluates the model on the training and test data respectively
    1. Predictions on test data
    2. Classification report
    3. Confusion matrix
    4. ROC curve

    Inputs:
    y_test: numpy array with test labels
    y_pred: numpy array with predicted test labels
    """
    
    plot_matrix = False
    if y_pred.shape[1] < 102:
        plot_matrix = False
        
    
    if y_pred.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    
    # Confusion matrix
    # Create a confusion matrix of the test predictions
    cm = confusion_matrix(y_test, y_pred)
    print(f"cm{cm}")
        # create heatmap
        # Set the size of the plot
    if plot_matrix:
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
        # Set plot labels
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        # Display plot
        plt.show()
    '''
    #create ROC curve
    from sklearn.preprocessing import LabelBinarizer
    fig, ax = plt.subplots(figsize=(15, 15))

    label_binarizer = LabelBinarizer().fit(y_test)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_pred = label_binarizer.transform(y_pred)
    
    if (y_onehot_pred.shape[1] < 2):
        fpr, tpr, _ = roc_curve(y_test,  y_pred)

        #create ROC curve
        #plt.plot(fpr,tpr)
        if y_prob is not None:
            RocCurveDisplay.from_predictions(
                    y_test,
                    y_prob,
                    name=f"ROC curve",
                    color='aqua',
                    ax=ax,
                )
        else:
            RocCurveDisplay.from_predictions(
                    y_test,
                    y_pred,
                    name=f"ROC curve",
                    color='aqua',
                    ax=ax,
                )
        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        
    else:
        from itertools import cycle
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "yellow", "purple", "pink", "brown", "black"])
        if y_prob is None:
            for class_id, color in zip(range(len(label_binarizer.classes_)), colors):
                RocCurveDisplay.from_predictions(
                    y_onehot_test[:, class_id],
                    y_onehot_pred[:, class_id],
                    name=f"ROC curve for {label_binarizer.classes_[class_id]}",
                    color=color,
                    ax=ax,
                )
        else:
            for class_id, color in zip(range(len(label_binarizer.classes_)), colors):
                RocCurveDisplay.from_predictions(
                    y_onehot_test[:, class_id],
                    y_prob[:, class_id],
                    name=f"ROC curve for {label_binarizer.classes_[class_id]}",
                    color=color,
                    ax=ax,
                )

        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
        plt.show()
    '''
    # Classification report
    # Create a classification report of the test predictions
    cr = classification_report(y_test, y_pred)
    # print classification report
    print(cr)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class precision
    recall = recall_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class recall
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class F1-score
    ba = balanced_accuracy_score(y_test, y_pred)
    print("F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format(c=f1, d = accuracy, e= ba))
   
    return accuracy, precision, recall, f1


def test(model, test_dataloader, saliency=True, device='cpu', save=False):

    model.to(device)
    model.eval()

    output_size = test_dataloader.dataset.labels.shape[1]
    num_classes = 2 if test_dataloader.dataset.labels.shape[1] == 1 else test_dataloader.dataset.labels.shape[1]

    eval_images_per_class = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        y_true, y_pred = [], []
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            image, labels =  batch['image'], batch['labels'].to(device)
            if image is not None:
                if type(image) is list:
                    for idx in range(len(image)):
                        image[idx] = image[idx].to(device)
                else:
                    image = image.to(device)
            if 'meta' in batch:
                meta = batch['meta'].to(device)
                outputs = model(image, meta)
            else:
                outputs = model(image)            

            if (output_size == 1):
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Get 5 images per class for saliency maps
      #      for i in range(num_classes):
      #          if len(eval_images_per_class[i]) < 5:
      #              if (output_size == 1):
      #                  eval_images_per_class[i] += [img for i, img in enumerate(image) if labels[i] == i]
      #              else:
      #                  eval_images_per_class[i] += [img for i, img in enumerate(image) if np.argmax(labels[i].cpu().numpy()) == i]
                    
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        if (output_size == 1):
            y_pred_one_hot = (y_pred > 0.5).astype(int)
        else:
            predicted_class_indices = np.argmax(y_pred, axis=1)
            # Convert the predicted class indices to one-hot encoding
            y_pred_one_hot = np.eye(y_pred.shape[1])[predicted_class_indices]
            
        # If the output size is 2, then we just need the probabilities of the positive class
        if (output_size == 2):
            y_pred = y_pred[:, 1]
        #print(y_pred)
        test_model(y_true, y_pred_one_hot, y_pred)
        #print(y_true)
        #print(y_pred)
        return y_pred, y_true
  
    if saliency:
        if save:
            os.makedirs('saliency_maps', exist_ok=True)
        
        print('#' * 50, f' Saliency Maps ', '#' * 50)
        print('')

        # Select some evaluation images to generate saliency maps
        eval_images = []
        for img_class in eval_images_per_class.keys():
            eval_images = eval_images_per_class[img_class][:5]

            print(f'Class {img_class}:')
            i = 0
            for eval_image in eval_images:
                eval_image = eval_image.unsqueeze(0)  # Add batch dimension
                saliency_map = get_saliency_map(model, eval_image)

                # Plot original image and saliency map side by side
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(eval_image[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.title(f'Original Image (Class {img_class})')
                
                plt.subplot(1, 2, 2)
                plt.imshow(saliency_map[0].detach().cpu().numpy(), cmap=plt.cm.hot)
                plt.title('Saliency Map')
                
                plt.tight_layout()
                if save:
                    plt.savefig(f'saliency_maps/saliency_map_class_{img_class}_image_{i}.pdf')
                    i+=1
                    
                plt.show()
def test_multitask(model, task_index, test_dataloader, criterion_list, saliency=True, device='cpu', save=False):

    model.to(device)
    model.eval()

    #output_size = test_dataloader.dataset.labels.shape[1]
    #num_classes = 2 if test_dataloader.dataset.labels.shape[1] == 1 else test_dataloader.dataset.labels.shape[1]

    #eval_images_per_class = {i: [] for i in range(num_classes)}
    

    all_preds = [[],[],[],[],[],[],[],[],[]]
    all_soft_preds = [[],[],[],[],[],[],[],[],[]]
    all_labels = [[],[],[],[],[],[],[],[],[]]
    avg_test_loss = 0.0
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, total=len(test_dataloader)):
            test_inputs = test_batch['image']
            test_labels = test_batch['labels'].to(device)
            for idx in range(len(test_inputs)):
                test_inputs[idx] = test_inputs[idx].to(device)
            meta = test_batch['meta'].to(device)
            test_outputs = model(test_inputs, meta)
            for i in range(min(len(criterion_list),6)): #change to 6 !!!
                if i in task_index:
                    loss = criterion_list[i](test_outputs[i], test_labels[:,i])
                    avg_test_loss += loss.item()
                preds = torch.argmax(test_outputs[i], dim=1) 
                soft_preds = torch.softmax(test_outputs[i],dim=1)
                
                all_preds[i].extend(preds.cpu().numpy())
                all_labels[i].extend(test_labels[:,i].cpu().numpy())
                all_soft_preds[i].extend(soft_preds[:,1].cpu().numpy())
        
         
    avg_test_loss /= len(test_dataloader)
        
        
    for i in range(min(len(criterion_list),6)):
        f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
        acc = accuracy_score(all_labels[i], all_preds[i])
        ba = balanced_accuracy_score(all_labels[i], all_preds[i])
        confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
        print("Test loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_test_loss, c=f1, d = acc, e= ba))
        print(f"cm{confusion_matrix_sc}")


    return all_soft_preds, all_labels

    
    if saliency:
        if save:
            os.makedirs('saliency_maps', exist_ok=True)
        
        print('#' * 50, f' Saliency Maps ', '#' * 50)
        print('')

        # Select some evaluation images to generate saliency maps
        eval_images = []
        for img_class in eval_images_per_class.keys():
            eval_images = eval_images_per_class[img_class][:5]

            print(f'Class {img_class}:')
            i = 0
            for eval_image in eval_images:
                eval_image = eval_image.unsqueeze(0)  # Add batch dimension
                saliency_map = get_saliency_map(model, eval_image)

                # Plot original image and saliency map side by side
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(eval_image[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.title(f'Original Image (Class {img_class})')
                
                plt.subplot(1, 2, 2)
                plt.imshow(saliency_map[0].detach().cpu().numpy(), cmap=plt.cm.hot)
                plt.title('Saliency Map')
                
                plt.tight_layout()
                if save:
                    plt.savefig(f'saliency_maps/saliency_map_class_{img_class}_image_{i}.pdf')
                    i+=1
                    
                plt.show()


def test_joint_multitask(model, task_index, test_dataloader, criterion_list1,criterion_list2, saliency=True, device='cpu', save=False):

    model.to(device)
    model.eval()

    #output_size = test_dataloader.dataset.labels.shape[1]
    #num_classes = 2 if test_dataloader.dataset.labels.shape[1] == 1 else test_dataloader.dataset.labels.shape[1]

    #eval_images_per_class = {i: [] for i in range(num_classes)}
    

    all_preds = [[],[],[],[],[],[],[],[],[]]
    all_soft_preds = [[],[],[],[],[],[],[],[],[]]
    all_labels = [[],[],[],[],[],[],[],[],[]]
    avg_test_loss = 0.0
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, total=len(test_dataloader)):
            test_inputs = test_batch['image']
            test_labels_full = test_batch['labels'].to(device)
            test_labels1 = test_labels_full[:,:4]
            test_labels2 = test_labels_full[:,4:]
            for idx in range(len(test_inputs)):
                test_inputs[idx] = test_inputs[idx].to(device)
            meta = test_batch['meta'].to(device)
            test_outputs1,test_outputs2 = model(test_inputs, meta)
            for i in range(min(len(criterion_list1),6)): #change to 6 !!!
                if i in task_index:
                    loss = criterion_list1[i](test_outputs1[i], test_labels1[:,i])
                    avg_test_loss += loss.item()
                    loss = criterion_list2[i](test_outputs2[i], test_labels2[:,i])
                    avg_test_loss += loss.item()

                preds1 = torch.argmax(test_outputs1[i], dim=1) 
                soft_preds1 = torch.softmax(test_outputs1[i],dim=1)
                
                all_preds[i].extend(preds1.cpu().numpy())
                all_labels[i].extend(test_labels1[:,i].cpu().numpy())
                all_soft_preds[i].extend(soft_preds1[:,1].cpu().numpy())
                preds2 = torch.argmax(test_outputs2[i], dim=1) 
                soft_preds2 = torch.softmax(test_outputs2[i],dim=1)
                
                all_preds[i].extend(preds2.cpu().numpy())
                all_labels[i].extend(test_labels2[:,i].cpu().numpy())
                all_soft_preds[i].extend(soft_preds2[:,1].cpu().numpy())
        
         
    avg_test_loss /= len(test_dataloader)
        
        
    for i in range(min(len(criterion_list1),6)):
        f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
        acc = accuracy_score(all_labels[i], all_preds[i])
        ba = balanced_accuracy_score(all_labels[i], all_preds[i])
        confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
        print("Test loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_test_loss, c=f1, d = acc, e= ba))
        print(f"cm{confusion_matrix_sc}")


    return all_soft_preds, all_labels

    
    if saliency:
        if save:
            os.makedirs('saliency_maps', exist_ok=True)
        
        print('#' * 50, f' Saliency Maps ', '#' * 50)
        print('')

        # Select some evaluation images to generate saliency maps
        eval_images = []
        for img_class in eval_images_per_class.keys():
            eval_images = eval_images_per_class[img_class][:5]

            print(f'Class {img_class}:')
            i = 0
            for eval_image in eval_images:
                eval_image = eval_image.unsqueeze(0)  # Add batch dimension
                saliency_map = get_saliency_map(model, eval_image)

                # Plot original image and saliency map side by side
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(eval_image[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.title(f'Original Image (Class {img_class})')
                
                plt.subplot(1, 2, 2)
                plt.imshow(saliency_map[0].detach().cpu().numpy(), cmap=plt.cm.hot)
                plt.title('Saliency Map')
                
                plt.tight_layout()
                if save:
                    plt.savefig(f'saliency_maps/saliency_map_class_{img_class}_image_{i}.pdf')
                    i+=1
                    
                plt.show()

def test_DR(model, task_index, test_dataloader, criterion_list, saliency=True, device='cpu', save=False):

    model.to(device)
    model.eval()

    #output_size = test_dataloader.dataset.labels.shape[1]
    #num_classes = 2 if test_dataloader.dataset.labels.shape[1] == 1 else test_dataloader.dataset.labels.shape[1]

    #eval_images_per_class = {i: [] for i in range(num_classes)}
    

    all_preds = [[],[],[],[],[],[],[],[],[]]
    all_soft_preds = [[],[],[],[],[],[],[],[],[]]
    all_labels = [[],[],[],[],[],[],[],[],[]]
    avg_test_loss = 0.0
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, total=len(test_dataloader)):
            test_inputs = test_batch['image']
            test_labels = test_batch['labels'].to(device)
            for idx in range(len(test_inputs)):
                test_inputs[idx] = test_inputs[idx].to(device)
            meta = test_batch['meta'].to(device)
            test_outputs = model(test_inputs, meta)
            for i in range(min(len(criterion_list),6)): #change to 6 !!!
                if i in task_index:
                    loss = criterion_list[i](test_outputs[i], test_labels[:,i].float())
                    avg_test_loss += loss.item()
                preds = torch.round(test_outputs[i]).long()
               # soft_preds = torch.softmax(test_outputs[i],dim=1)
                
                all_preds[i].extend(preds.cpu().numpy())
                all_labels[i].extend(test_labels[:,i].cpu().numpy())
               # all_soft_preds[i].extend(soft_preds[:,1].cpu().numpy())
        
         
    avg_test_loss /= len(test_dataloader)
        
        
    for i in range(min(len(criterion_list),6)):
        f1 = f1_score(all_labels[i], all_preds[i], average='macro') #computing f1 score for one of two tasks for one epoch
        acc = accuracy_score(all_labels[i], all_preds[i])
        ba = balanced_accuracy_score(all_labels[i], all_preds[i])
        confusion_matrix_sc = confusion_matrix(all_labels[i], all_preds[i])
        print("Test loss {b:.3f}, F1 {c:.3f}, Acc {d:.3f}, BA {e:.3f}".format( b=avg_test_loss, c=f1, d = acc, e= ba))
        print(f"cm{confusion_matrix_sc}")


    return all_soft_preds, all_labels

    
    if saliency:
        if save:
            os.makedirs('saliency_maps', exist_ok=True)
        
        print('#' * 50, f' Saliency Maps ', '#' * 50)
        print('')

        # Select some evaluation images to generate saliency maps
        eval_images = []
        for img_class in eval_images_per_class.keys():
            eval_images = eval_images_per_class[img_class][:5]

            print(f'Class {img_class}:')
            i = 0
            for eval_image in eval_images:
                eval_image = eval_image.unsqueeze(0)  # Add batch dimension
                saliency_map = get_saliency_map(model, eval_image)

                # Plot original image and saliency map side by side
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(eval_image[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.title(f'Original Image (Class {img_class})')
                
                plt.subplot(1, 2, 2)
                plt.imshow(saliency_map[0].detach().cpu().numpy(), cmap=plt.cm.hot)
                plt.title('Saliency Map')
                
                plt.tight_layout()
                if save:
                    plt.savefig(f'saliency_maps/saliency_map_class_{img_class}_image_{i}.pdf')
                    i+=1
                    
                plt.show()
