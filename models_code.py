# to add in the future - use not only accuracy. 
# using different number of trees in random forest
# using different kernal in svm
# using different number of neighbors in knn
# use other max_iter num in logistic regression (first run: 100 iter reached max, didn't converge, lbfgs, 38.85% accuracy)
# use different solver - logistic regression
# (use different number of epochs in CNN)
# use different proportion of train/test split

# check and write CNN 
# get rid of image loading separate functions?
# add: dimensionality reduction, visualizations, statistics per class, ovetfitting check, data augmentation\masking?

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image  # Use PIL instead of tensorflow

# our own image loading functions
def load_img(img_path, target_size=None):
    """Load an image and optionally resize it to target_size"""
    img = Image.open(img_path)
    if target_size:
        img = img.resize(target_size)
    return img

def img_to_array(img):
    """Convert PIL Image to numpy array"""
    return np.array(img)



# if it's not the first run, load np arrays versions of data, save time
if os.path.exists('X_eurosat.npy') and os.path.exists('y_eurosat.npy'):
    print("a\n")
    X = np.load('X_eurosat.npy')
    y = np.load('y_eurosat.npy')
    print("b\n")

else:
    # Path to EuroSAT_RGB dataset
    data_dir = r'C:\Users\ayele\Documents\ML_2025\ML_project\EuroSAT_RGB'

    images = []
    labels = [] # will be a list the size of all dataset - for i - images[i] is of category labels[i]

    # load images and labels
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = load_img(img_path, target_size=(64, 64))  # EuroSAT images are 64x64
                img_array = img_to_array(img)
                images.append(img_array.flatten())
                labels.append(label)
    # plot data distribution
    # import matplotlib.pyplot as plt
    # from collections import Counter
    # Plot data distribution with counts on top of bars and a blue-green colormap
    # label_counts = Counter(labels)
    # plt.figure(figsize=(10, 5))
    # bars = plt.bar(label_counts.keys(), label_counts.values(), color=plt.cm.viridis(np.linspace(0, 1, len(label_counts))))
    # plt.xlabel('Class Label')
    # plt.ylabel('Number of Samples')
    # plt.title('Class Distribution in EuroSAT Dataset')
    # plt.xticks(rotation=45)
    # # Add counts on top of bars
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.annotate(f'{int(height)}',
    #                  xy=(bar.get_x() + bar.get_width() / 2, height),
    #                  xytext=(0, 3),  # 3 points vertical offset
    #                  textcoords="offset points",
    #                  ha='center', va='bottom')
    # plt.tight_layout()
    # plt.show()

    # plot images representing each class
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    # for i, label in enumerate(np.unique(labels)):
    #     class_images = [img for img, lbl in zip(images, labels) if lbl == label]
    #     if class_images:
    #         axes[i // 5, i % 5].imshow(class_images[0].reshape(64, 64, 3).astype(np.uint8))
    #         axes[i // 5, i % 5].set_title(label)
    #         axes[i // 5, i % 5].axis('off')
    # plt.tight_layout()
    # plt.show()


    X = np.array(images)
    le = LabelEncoder() # it changes labels from string to sequential numbers, easier for models to work with
    y = le.fit_transform(labels)
    # print a dictionary of numbers and the labels they represent
    label_dict = {i: label for i, label in enumerate(le.classes_)}
    print("Label dictionary:", label_dict)

    # save processed arrays - images and labels as np arrays (to save loading time every run)
    np.save('X_eurosat.npy', X)
    np.save('y_eurosat.npy', y)

# def data_statistics(X, labels):
#     """Print basic statistics about the dataset."""
#     print(f"Number of samples: {X.shape[0]}")
#     print(f"Number of features: {X.shape[1]}")
#     print(f"Number of classes: {len(np.unique(y))}")
#     print("Class distribution:")
#     unique, counts = np.unique(y, return_counts=True)
#     class_distribution = dict(zip(unique, counts))
#     for class_label, count in class_distribution.items():
#         print(f"Class {class_label}: {count} samples")
#     # plot class distribution
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 5))
#     plt.bar(class_distribution.keys(), class_distribution.values())
#     plt.xlabel('Class Label')
#     plt.ylabel('Number of Samples')
#     plt.title('Class Distribution')
#     plt.xticks(rotation=45)
#     plt.show()
# splitting data into training and testing sets, here 80% for training and 20% for testing - - try other as changes tryed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def logisticRegression(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    iter = 200  # max_iter for convergence
    solverr = 'saga'  # using 'lbfgs' solver vs 'saga' vs `newton-cg`
    from sklearn.decomposition import PCA  # added pca to speed up training
    pca = PCA(n_components=100)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Create and train the model
    model = LogisticRegression(solver=solverr, max_iter = iter)# different solver
    print(f"LR regular Model {solverr} solver, {iter} iterations started")
    model.fit(X_train, y_train)

    # Evaluate the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"LR Model {solverr} solver, {iter} iterations train accuracy: {train_accuracy * 100:.2f}%")
    print(f"LR Model {solverr} solver, {iter} iterations test accuracy: {test_accuracy * 100:.2f}%")
    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Logistic Regression Confusion Matrix')
    plt.show()
    


def randomForest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from datetime import datetime
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)  # reduce dimensionality to speed up training
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    n_trees = 400  # num of trees in the forest
    print("RM pca {n_trees} TREES Model started")
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42 )
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    with open('svm_output.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - RM 200 TREES Model accuracy: {accuracy * 100:.2f}%\n")
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'RF Confusion Matrix')
    plt.show()


def SVM(X_train, X_test, y_train, y_test):
    from sklearn.svm import LinearSVC  # Using LinearSVC for linear kernel
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime
    iter=200
    svm = LinearSVC(random_state=42, max_iter=iter ) # Using linear kernel - try other as changes tryed
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    comps =100
    print("SVM Model, {iter} iters, {comps} comps started")

    pca = PCA(n_components=comps)  # Reduce dimensionality to speed up training
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    model = make_pipeline(scaler, pca, svm)  # Create a pipeline with PCA and SVM
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    # print output to svm_output.txt file
    with open('svm_output.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - SVM Model, {comps} pca, {iter} iters, accuracy: {accuracy * 100:.2f}%\n")
    print(f"SVM {comps} PCA, {iter} iters, Model accuracy: {accuracy * 100:.2f}%")

    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'SVM + PCA Confusion Matrix')
    plt.show()

def KNN(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)  # Using 3/5 neighbors - try other as changes tryed
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'KNN Confusion Matrix')
    plt.show()
    

def adaboost(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from datetime import datetime
    from collections import Counter
    depth = 3  # max depth of the decision tree
    n_trees=200
    print("AdaBoost Model, {n_trees} trees, {depth} depth started")
    # add pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=85)  # Reduce dimensionality to speed up training
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    # Create and train the model
    # Using DecisionTreeClassifier as the base estimator
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=n_trees,  # Number of trees in the ensemble
        random_state=42)
    model.fit(X_train, y_train)  # y_train can have multiple classes
    # Evaluate the model
    # print loss and accuracy for train and test sets
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"AdaBoost Model train accuracy: {train_accuracy * 100:.2f}%")
    print(f"AdaBoost Model test accuracy: {test_accuracy * 100:.2f}%")    
    with open('adaboost_output.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AdaBoost Model, {n_trees} trees, {depth} depth, accuracy: {test_accuracy * 100:.2f}%\n")

    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'AdaBoost Confusion Matrix')
    plt.show()





# running models
# data_statistics(X, y)  # Print dataset statistics

# logisticRegression(X_train, X_test, y_train, y_test) # first run - Model accuracy: 38.85%
# randomForest(X_train, X_test, y_train, y_test) # first run 100 trees - Model accuracy: 69.04%
# SVM(X_train, X_test, y_train, y_test) # ran over 30 minutes, didn't finish
# KNN(X_train, X_test, y_train, y_test) # first run - Model accuracy: 34.44%
# CNN(X_train, X_test, y_train, y_test) # problem with downloading tensorflow or pytorch, didn't run yet
# adaboost(X_train, X_test, y_train, y_test) # first run 10 trees- Model accuracy: 25.50
# print("hi")


import matplotlib.pyplot as plt

# # Example data
# # Number of trees (estimators)
# trees = [70, 100, 200, 70, 100, 200, 300, 400, 70, 100, 300, 400, 200]

# # Number of PCA components (None = no PCA, replaced '-' with 0)
# pca = [0, 0, 0, 50, 50, 50, 50, 50, 100, 100, 100, 100, 200]

# # Accuracy
# accuracy = [68.43, 69.04, 69.56, 64.57, 65.15, 66.81, 66.63, 66.76, 64.94, 65.20, 67.13, 68.00, 66.30]
# Max iterations (max_iter) - SVM
# max_iter = [100, 100, 200, 300, 200, 200, 300, 300]

# # PCA components
# pca =      [50,  100, 200, 300, 50,  100, 100, 50]

# # Accuracy
# accuracy = [38.94, 39.13, 39.07, 38.63, 38.93, 39.24, 39.02, 38.85]

# plt.scatter(max_iter, pca, c=accuracy, cmap='Reds', s=100)  # s is size of dots

# # plt.scatter(trees, pca, c=accuracy, cmap='Reds', s=100)  # s is size of dots
# plt.colorbar(label='accuracy (color scale)')
# plt.xlabel("trees Axis")
# plt.ylabel("pca Axis")
# plt.title("SVM Trees vs. PCA Scatter Plot (Accuracy as Color)")
# # plt.title("RF Trees vs. PCA Scatter Plot (Accuracy as Color)")
# plt.grid(True)
# plt.show()

# running resnet on the dataset - finetuning a pre-trained reset18
# import torch
# import torchvision
# from torchvision import datasets, transforms
# from torchvision.models import resnet18
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score
# def train_resnet():
# #     # Define transformations for the training and validation sets
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),  # Resize images to 64x64

#         transforms.ToTensor(),  # Convert images to PyTorch tensors
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
#     ])
# #     # Load the EuroSAT dataset
#     data_dir = r'C:\Users\ayele\Documents\ML_2025\ML_project\EuroSAT_RGB'
#     dataset = datasets.ImageFolder(root=data_dir, transform=transform)
#     train_size = int(0.8 * len(dataset))  # 80% for training
#     val_size = len(dataset) - train_size  # 20% for validation
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#     # Load a pre-trained ResNet model
#     model = resnet18(pretrained=True)  # Load pre-trained ResNet-18
#     # Modify the final layer to match the number of classes in EuroSAT
#     num_classes = len(dataset.classes)  # Number of classes in EuroSAT
#     model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace the final layer
#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
#     optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
#     # Training loop
#     num_epochs = 10  # Number of epochs to train
#     for epoch in range(num_epochs):
#         model.train()  # Set the model to training mode
#         running_loss = 0.0
#         for images, labels in train_loader:
#             optimizer.zero_grad()  # Zero the gradients
#             outputs = model(images)  # Forward pass
#             loss = criterion(outputs, labels)  # Compute loss
#             loss.backward()  # Backward pass
#             optimizer.step()  # Update weights
#             running_loss += loss.item()  # Accumulate loss
#         avg_loss = running_loss / len(train_loader)  # Average loss for the epoch
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
#     # Validation
#     model.eval()  # Set the model to evaluation mode
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():  # No gradient computation during validation
#         for images, labels in val_loader:
#             outputs = model(images)  # Forward pass
#             _, preds = torch.max(outputs, 1)  # Get predicted class indices
#             all_preds.extend(preds.cpu().numpy())  # Store predictions
#             all_labels.extend(labels.cpu().numpy())  # Store true labels
#     # Calculate accuracy
#     accuracy = accuracy_score(all_labels, all_preds)  # Compute accuracy
#     print(f"Validation Accuracy: {accuracy * 100:.2f}%")
#     return model  # Return the trained model
# # model = train_resnet()  # Call the function to train the ResNet model
# # # save the model
# # import torch
# # torch.save(model.state_dict(), 'resnet_eurosat.pth')  # Save the model state dictionary
# # # load the model
# # model = resnet18()  # Initialize a new ResNet-18 model
# # model.fc = nn.Linear(model.fc.in_features, 10)  # Modify the final layer
# # model.load_state_dict(torch.load('resnet_eurosat.pth'))  # Load the saved state
# # model.eval()  # Set the model to evaluation mode
# # print("ResNet model loaded and ready for inference.")

# # Recreate model architecture
# model = resnet18()
# model.fc = nn.Linear(model.fc.in_features, 10)

# # Load the trained weights
# model.load_state_dict(torch.load('resnet_eurosat.pth'))

# # Set to evaluation mode
# model.eval()

# print("✅ ResNet model loaded and ready for inference.")
