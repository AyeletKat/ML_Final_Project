# to add in the future - use not only accuracy. 
# using different number of trees in random forest
# using different kernal in svm
# using different number of neighbors in knn
# use other max_iter num in logistic regression (first run: 100 iter reached max, didn't converge, lbfgs, 38.85% accuracy)
# use different solver - logistic regression
# (use different number of epochs in CNN)
# use different proportion of train/test split

# add: dimensionality reduction, visualizations, statistics per class, overfitting check, data augmentation/masking?

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

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
    data_dir = r'C:\Users\ayele\Documents\ML_2025\ML_project\EuroSAT_RGB' # change to your path (where you downloaded the dataset)

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

    X = np.array(images)
    le = LabelEncoder() # it changes labels from string to sequential numbers, easier for models to work with
    y = le.fit_transform(labels)

    # save processed arrays - images and labels as np arrays (to save loading time every run)
    np.save('X_eurosat.npy', X)
    np.save('y_eurosat.npy', y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def logisticRegression(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    iter = 200  # max_iter for convergence
    solverr = 'newton-cg'  # Using 'lbfgs' solver, can try others like 'saga' `newton-cg`
    # add pca to speed up training
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=100)  # Reduce dimensionality to speed up training
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    # Create and train the model
    model = LogisticRegression(solver=solverr, max_iter = iter)# different solver
    print(f"LR regular Model {solverr} solver, {iter} iterations started")
    model.fit(X_train, y_train)

    # Evaluate the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"LR Model {solverr} solver, {iter} iterations train accuracy: {train_accuracy * 100:.2f}%")
    print(f"LR Model {solverr} solver, {iter} iterations test accuracy: {test_accuracy * 100:.2f}%")


def randomForest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from datetime import datetime
    # add pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)  # Reduce dimensionality to speed up training
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    n_trees = 400  # Number of trees in the forest
    print("RM pca {n_trees} TREES Model started")
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42 )
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    with open('svm_output.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - RM 200 TREES Model accuracy: {accuracy * 100:.2f}%\n")
    print(f"Model accuracy: {accuracy * 100:.2f}%")


def SVM(X_train, X_test, y_train, y_test):
    from sklearn.svm import LinearSVC  # Using LinearSVC for linear kernel
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime
    iter=300
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


def KNN(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=13)  # Using 3/5 neighbors - try other as changes tryed
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

def adaboost(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from datetime import datetime
    from collections import Counter
    depth = 3  # max depth of the decision tree
    n_trees=50
    print("AdaBoost Model, {n_trees} trees, {depth} depth started")

    # avg_training_loss = 0.0  # Initialize average training loss
    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=n_trees,  # Number of trees in the ensemble
        random_state=42)
    model.fit(X_train, y_train)  # y_train can have multiple classes
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    with open('adaboost_output.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AdaBoost Model, {n_trees} trees, {depth} depth, accuracy: {accuracy * 100:.2f}%\n")
    print(f"AdaBoost Model accuracy: {accuracy * 100:.2f}%")



# running models
# data_statistics(X, y)  # Print dataset statistics

logisticRegression(X_train, X_test, y_train, y_test) # first run - Model accuracy: 38.85%
# randomForest(X_train, X_test, y_train, y_test) # first run 100 trees - Model accuracy: 69.04%
# SVM(X_train, X_test, y_train, y_test) # ran over 30 minutes, didn't finish
# KNN(X_train, X_test, y_train, y_test) # first run - Model accuracy: 34.44%
# adaboost(X_train, X_test, y_train, y_test) # first run 10 trees- Model accuracy: 25.50
