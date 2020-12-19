from PIL import Image
import numpy as np
import tensorflow as tf
import os
import sys
import cv2
from sklearn.svm import NuSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
import time
import traceback
import bayes_optimization
from sklearn.model_selection import GridSearchCV

data_path = 'dataset/'


def generate_data():
    covid_data = os.listdir('dataset/CT_COVID')
    non_covid_data = os.listdir('dataset/CT_NonCOVID')
    x_data = []
    y_data = []
    #Add positive covid data
    for fileName in covid_data:
        #im = Image.open(data_path + 'CT_COVID/' + fileName).convert("RGB")
        im = cv2.imread(data_path + 'CT_COVID/' + fileName, cv2.IMREAD_COLOR)
        resize = (224,224)
        #im = im.resize(resize)
        im = cv2.resize(im, resize)
        img_arr = np.array(im)
        x_data.append(img_arr/255)      #normalize data
        y_data.append(1)
    
    #Add negative covid data
    for fileName in non_covid_data:
        #im = Image.open(data_path + 'CT_NonCOVID/' + fileName).convert("RGB")
        im = cv2.imread(data_path + 'CT_NonCOVID/' + fileName, cv2.IMREAD_COLOR)
        resize = (224,224)
        #im = im.resize(resize)
        im = cv2.resize(im, resize)
        img_arr = np.array(im)
        x_data.append(img_arr/255)      #normalize data
        y_data.append(0)
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print (x_data.shape)
    print (y_data.shape)
    return x_data, y_data

def get_model(modelName):
    global output_size
    # Create the model.
    model_input = tf.keras.Input(shape=(224, 224, 3))

    # Try out different models
    if modelName == 'DenseNet':
        model = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3))
        output_size = 1024
    elif modelName == 'InceptionV3':
        model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))
        output_size = 2048
    elif modelName == 'ResNet50V2':
        model = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=(224,224,3))
        output_size = 2048
    elif modelName == 'ResNet50V1':
        model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
        output_size = 2048
    elif modelName == 'MobileNetV1':
        model = tf.keras.applications.MobileNet(include_top=False, weights='imagenet', input_shape=(224,224,3))
        output_size = 1024
    elif modelName == 'MobileNetV2':
        model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
        output_size = 1280
    
    model_output = model(model_input)
    model_output = tf.keras.layers.GlobalAveragePooling2D(name='ga')(model_output)
    feature_model = tf.keras.models.Model(inputs=model_input, outputs=model_output)
    return feature_model

def extract_features(x, feature_model):
    global output_size
    temp = []
    x_samples = x.shape[0]
    for i in range(x_samples):
        temp.append(feature_model(np.array([x[i]])))
    temp = np.array(temp)
    return temp.reshape(x_samples, output_size)

def evaluate_model(classifier, x, y):
    # Using StratifiedKfold cross validation with splits = 10
    cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=603)

    accuracy = cross_val_score(classifier, x, y, cv=cross_validation)
    scores = cross_val_score(classifier, x, y, cv=cross_validation, scoring='roc_auc')
    recall = cross_val_score(classifier, x, y, cv=cross_validation, scoring='recall')
    precision = cross_val_score(classifier, x, y, cv=cross_validation, scoring='precision')
    f1 = (2 * recall * precision) / (precision + recall)

    print (f'Accuracy: {np.mean(accuracy)} + {np.std(accuracy)}')
    print (f'Area Under Curve: {np.mean(scores)} + {np.std(scores)}')
    print (f'Recall: {np.mean(recall)} + {np.std(recall)}')
    print (f'Presicion: {np.mean(precision)} + {np.std(precision)}')
    print (f'f1 score: {np.mean(f1)} + {np.std(f1)}')

def predict(feature_model, classifier):
    global output_size
    print ("Please provide test image relative file path in png/jpg format")
    inp = input()
    im = Image.open(inp).convert('RGB')
    resize = (224, 224)
    im = im.resize(resize)
    img_arr = np.array(im)
    img_arr = img_arr/255
    fx = feature_model(np.array([img_arr]))
    fx = np.array(fx).reshape(output_size, 1)
    pred = np.round(classifier.predict(fx.reshape(1, -1)))
    if pred == 1:
        print ("Covid diagnosis: {}".format(True))
    else:
        print ("Covid diagnosis: {}".format(False))

if __name__ == "__main__":
    global output_size
    try:
        #data-preprocessing
        print ("Loading data...")
        x, y = generate_data()
        print ("Data loading and pre-processing done.")
        #params = bayes_optimization.bayesOpt(x, y)
        #print(params)
        #get model
        model_dict = {
            1: "DenseNet",
            2: "InceptionV3",
            3: "ResNet50V2",
            4: "ResNet50V1",
            5: "MobileNetV1",
            6: "MobileNetV2"
        }
        print ("Enter model type option between 1-5.")
        print ("Enter 1 for DenseNet.")
        print ("Enter 2 for InceptionV3.")
        print ("Enter 3 for ResNet50V2.")
        print ("Enter 4 for ResNet50V1.")
        print ("Enter 5 for MobileNetV1.")
        print ("Enter 6 for MobileNetV2.")
        model_option = input()
        modelName = model_dict.get(int(model_option))

        print ("Initializing feature model...")
        feature_model = get_model(modelName)     #tensorflow denseNet121, InceptionV3, ResNet50V2, ResNet50V1, MobileNetV1
        #set nu-SVM classifier with optimal params from paper
        print ("Initializing classifier...")
        # defining parameter range 
        #param_grid = {'nu': [0.5],  
                    #'gamma': np.arange(0.009, 0.01+0.0000, 0.0001).tolist(),
                    #'max_iter': [163],
                    #'kernel': ['rbf']} 
  
        #classifier = GridSearchCV(NuSVC(), param_grid, cv = 5, verbose=3, refit=True, n_jobs=-1)
        #print( classifier.best_score_ )
        #print( classifier.best_params_ )
        #extract features and reorgranize X,Y data
        print ("Extracting features...")
        x = extract_features(x, feature_model)      #replacing with features computed from deep learning model
        classifier = NuSVC(nu=0.4, kernel='rbf', gamma=0.0098, shrinking=True, tol=0.00001,
          max_iter=163, random_state=603, class_weight='balanced', probability=True)
        
        #train and fit
        print ("Training started...")
        classifier.fit(x, y)

        #print (classifier.best_score_)
        #print (classifier.best_params_)

        #evaluate classifier model on current dataset using K fold cross validation technique
        print ("Evaluating model performance...")
        evaluate_model(classifier, x, y)

        #test model with test data covid prediction
        while True:
            predict(feature_model, classifier)

    except KeyboardInterrupt:
        print ('Ctrl+C detected...Exiting')
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()
        print ('Exception occurred.... exiting')



