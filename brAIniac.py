# Imports Used

"""
OS - for file system operations and path checks on local machine
RANDOM - for reproducibility (randomness != reproducibility). reproducibility is to geneate same results repeatedly 
NUMPY - for array math
TENSORFLOW - backbone for keras and input pipelines without which keras can't optimize properly
MATPLOTLIB.PYLOT - To show images on a plot graph GUI
"""
import os, random, numpy as np, tensorflow as tf , matplotlib.pyplot as plt 

# To define our Convolutional Neural Network model and for creating augumented pipeline for our images
from keras.models import Sequential  
  
# To load our image data from local directory as datasets
from keras.utils import image_dataset_from_directory

"""
LAYERS - to access layers.Rescalling class for image normalization and to access data augmentation layers
METRICS - to access metrics such as Precision and Recall
"""
from keras import layers, metrics

#?????????????????????????????????? NOT USED NOW  ---- TO provide the Adam optimizer class
from keras.optimizers import Adam     

# To import the traditional CNN layers and classifiers
from keras.layers import Conv2D, MaxPooling2D , Flatten , Dense, Dropout

# for additional evaluation aside the keras in built metrics
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Creating the cnn model globally so that all functions can assess it wherever necessary
cnn_model= Sequential()

# global seed value to ensure reproducible randoms irrespective of how many times the script is ran  - 2025 thanks to BRISC2025
SEED = 2025

# apply the SEEDS to the random generators being used in this script to effect in any function either is being used
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# direct tensorflow to choose good configs for parallelism and prefetching based on local CPU, GPU and current workloads
AUTOTUNE = tf.data.AUTOTUNE

# data preparation function - to search local storage and fetch image data from local machines
def data_preparation():
    try:
        print("=" * 100)
        print("Fetching and preparing the Data...")

		
        computer_name = "chauk"         # <------ change to your pc name
        BASE_DIRECTORY = f"C:/Users/{computer_name}/Downloads/brisc2025/classification_task"
        
        if not os.path.exists(BASE_DIRECTORY):
            raise FileNotFoundError("Classification folder was not found!!!")
        
        # if found the append paths for train data and test data
        TRAIN_DIRECTORY = os.path.join(BASE_DIRECTORY, "train")
        TEST_DIRECTORY = os.path.join(BASE_DIRECTORY, "test")
        
        if not os.path.exists(TRAIN_DIRECTORY):
            raise FileNotFoundError("Train folder was not found!!!")
        
        if not os.path.exists(TEST_DIRECTORY):
            raise FileNotFoundError("Test folder was not found!!!")

		# Ensure all images will be resized to the images as in BRISC2025 raw data
        IMAGE_SIZE = (224,224)

		# set the amount of images to be used to train per step at a time to ensure we utilize CPU more effciently
        BATCH_SIZE = 32

		# names of tumors according to the folders in alphabetical order
        CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

		# information to be returned by the function
        data_preparation_details = {
            "BASE_DIRECTORY" : BASE_DIRECTORY,
            "TRAIN_DIRECTORY" : TRAIN_DIRECTORY,
            "TEST_DIRECTORY": TEST_DIRECTORY,
            "IMAGE_SIZE" : IMAGE_SIZE,
            "BATCH_SIZE" : BATCH_SIZE,
            "SEED" : SEED,
            "CLASS_NAMES" : CLASS_NAMES
        }
        
        print("Fetching and preparing task done")
        print("=" * 100)
        
        return data_preparation_details
    
    except FileNotFoundError as e:
        print ("Error message:  ", e)
        return None
    except:
        print("An unexpected error in data preparation function")
        return None

# data loading and preprocessing function 
def data_loading_and_preprocessing(data_preparation_details):
    try:
        print("=" * 100)
        print("Loading Data ...")
        
		# collect neccessary info from data preparation function result
        TRAIN_DIRECTORY = data_preparation_details["TRAIN_DIRECTORY"]
        TEST_DIRECTORY = data_preparation_details["TEST_DIRECTORY"]
        IMAGE_SIZE = data_preparation_details["IMAGE_SIZE"]
        BATCH_SIZE = data_preparation_details["BATCH_SIZE"]
        SEED = data_preparation_details["SEED"]

        # image_dataset_from directory to replace image_data_generator as it is currently deprecated
		# create training dataset by using the function image_dataset_from_directory
        training_dataset = image_dataset_from_directory (
            TRAIN_DIRECTORY,               	# directory to train data
            labels="inferred",          	# infer the labels and truths from the class folders
            label_mode="categorical",      	# to create one - hot label encodes styles such as [0,1,0,0] - representing an image is from meningioma class
            color_mode="rgb",               # set the color format
            image_size= IMAGE_SIZE,         # resize all images
            batch_size=BATCH_SIZE,         	# no of images per batch
            shuffle=True,                 	# randomize image order
            seed=SEED,                     	# ensure the shuffle randomly is reproducible
            validation_split= 0.1,        	# reserve 10% of the data is reserved for validation  
            subset="training"             	# this current dataset is for training not the validation
        )            

        # create the valiation dataset - to act as a pop quiz after every epoch of training
        validation_dataset = image_dataset_from_directory (
            TRAIN_DIRECTORY, 
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb", 
            image_size= IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,                   # for steady validation batches every run - like to compare performances against the same set of pop quizzes    
            seed=SEED,
            validation_split= 0.1,		
            subset="validation"          	 # this subset is for validation so the reserved 10 %           
        )

        """
        NOTE
        Over fitting happens when the models learns to much of the train data thereby memorizing it and performs badly on unseen data
        
        the validation set 90/10 is fine --- can help us check overfitting early on in training so implement a training stop if it happens
        
        Underfitting is seen when - when train and val accuarcies are low e.g 40/38, 55/50, 60/57  -> as epochs increase
        good fit - train and val acurracy are high and close to each other e.g 40/38, 70/68, 85/83
        overfitting - train accuracy is high , why validation accuracy is reducing e.g 50/48, 80/75, 90/70, 98/60
        
        causes of underfitting - not enough layers in model, smll no of neurons, few epochs, learning rate too low, bad data
        causes of overfitting - too many epochs, too many layers, not enough regularization like drop 10%, poor data augmentation
        """

        # create the testing dataset
        testing_dataset = image_dataset_from_directory (
            TEST_DIRECTORY, 
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb", 
            image_size= IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False                   
        )
       
        
        # pack data to return from function
        preprocessed_data = {
            "TRAINING_DATASET" : training_dataset,
            "VALIDATION_DATASET" : validation_dataset,
            "TESTING_DATASET" : testing_dataset
        }
        
        print("Data has been loaded")
        print("=" * 100)

        return preprocessed_data
    
    except:
        print("Unexpected error in Data loading and preprocessing function")
        return None


# data augmentation function 
def data_augmentation(preprocessed_data):
    try:       
        print("=" * 100)
        print("Augmenting imaes")

		# collect neccessary info from previous function
        training_dataset = preprocessed_data["TRAINING_DATASET"]
        validation_dataset = preprocessed_data["VALIDATION_DATASET"]
        testing_dataset = preprocessed_data["TESTING_DATASET"]

        """
        normalization algorithm for image pixel colors from range of 0-255 to 0-1
        Apply to every image so that the model focuses on patterns not magnitude of pixel values
        """
        rescale_pixels = layers.Rescaling(1./255)

        """
        create a sequential model to create an augmentation alogrithm for our dataset pipelines
        using layers from keras to create augmentation blocks
        """
        data_augmentation = Sequential (
            [
                layers.RandomContrast(0.1),       
                layers.RandomRotation(0.05),            
                layers.RandomZoom(0.1),               
                layers.RandomTranslation(0.05, 0.05)   
            ],
            name= "augment"                             
        )

		# apply the augmentation and normalization to training dataset
        training_dataset = training_dataset.map(
            lambda x, y :    (
                # applying augmentation to data
                # applied normalization and augmentation to images in this dataset || y is the label so remains untouched
                rescale_pixels(data_augmentation(x)),  y
            )    
        )

        # keep preprocessde data in ram for easy fetching and fine tunes performance using autotune
        """
        To imporove performance
        cache is used to store the current version of the transformed data in RAM for faster retrieval
        prefetch - will ensure that while model is trainign on another batch the AUTOTUNE from tensorflow perpares the next batch for training 
        """
        training_dataset = training_dataset.cache().prefetch(AUTOTUNE)       

		# just normalize images in validation set for stable validation set
        validation_dataset = validation_dataset.map(lambda x, y : (rescale_pixels(x)  , y))
        validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)   

        testing_dataset = testing_dataset.map(lambda x, y : (rescale_pixels(x)  , y))
        testing_dataset = testing_dataset.cache().prefetch(AUTOTUNE)      
        
        # data to be returned from this function
        augumented_data = {
            "TRAINING_DATASET" : training_dataset,
            "VALIDATION_DATASET" : validation_dataset,
            "TESTING_DATASET" : testing_dataset
        }    
       
        print("Augmentation done")
        print("=" * 100)
        
        return augumented_data
    
    except:
        print("Unexpected error in data augmentation function")
        return None  


# developing the cnn model function
def develop_cnn_model(data_preparation_details):
    try :

        print("=" * 100)
        print("Developing the Convolutional Neural Network model...")

        # ensure the global cnn_model is being used in this function
        global cnn_model

        # collect necessary data from data preparation function
        CLASS_NAMES = data_preparation_details["CLASS_NAMES"]

		# set the number of pattern detectors/lenses used needed to learn from each image
        filters = 32

        """
        Create 3 blocks of Convolution layer and Pooling layer each
        Each block increase the complexity of features the model learns
        from basic level of edges, corners, brigtness
        to mid level features such as textures, small shapes, curves
        to high level abstract features
        Each block learns from what previous blocks learned
        """

        # CNN BLOCK 1
        """
        Convolution layer - to learn features using pattern detectors
        (3,3) - the patch on image each filter views at a time
        input_shape - ensures that image sizes on first block are set with the 3 for the chanells of rgb
        relu activators - ensure non-linear learning pattern and keeps useful features
        padding same - adds zeros/blanks to be added around the edges of the image so that the filter can center itself even on edges and learn any patterns or features that may exist there. that way edges get the same attentions as center
        Convolution layer outputs 2D feature maps like cordinates of the pixels from the image height v width
        """
        cnn_model.add(Conv2D(filters, (3,3) , input_shape = (224,224,3), padding="same", activation = "relu"))
        
        # collects all features learnt previously and discarding unimportant areas of images by reducing it by half both on height and width
        cnn_model.add(MaxPooling2D(pool_size = (2, 2) ))
                
        # CNN BLOCK 2
        # double the pattern detectors to sompensate for the loss of spatial detail in maxpooling - the image is smaller and more complex patterns will emerge so more detectors to properly analyze it now
        filters *= 2    
        cnn_model.add(Conv2D(filters, (3,3), padding="same", activation = "relu"))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2) ))
        
        # CNN BLOCK 3
        filters *= 2   
        cnn_model.add(Conv2D(filters, (3,3), padding="same", activation = "relu"))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2) ))

        # flatten layer to convert the feature maps from 2D convo blocks to 1D vector that dense layer needs - basically summarzing the features from the convo block
        cnn_model.add(Flatten())

        """
        Dense layer aka fully connectional model collects summarized ID feature vector from flatten layer to analyse. it is acting as the brain
        Units 128 - acts as no of neurons that each reviews the feature vectors and adds biases (like important features vs unnecessary features) and grades them by weights(which are randomly assigned to different neurons). The larger the weights the more importance is placed on that feature - like a panel of jurors. 
        They collected translations of each neuron is evaluated in a weighted sum before passing their final collective report to the relu system which basically reads the summary. After reading what the neurons gave as final report , it checks against the actual truth and evaluates percentage accuracy
        If the accuracy is poor, it means that model perofrormed poorly so it switches up their weight values and tries again in process called backpropagation
        """
        cnn_model.add( Dense (units = 128, activation = "relu") )

        """
        Dropout layer - used to drop a percentage of the neurons - to avoid memorizing the data and causing overfitting - 
        Like dropping say 50% from a group of students that are always in same project group and asking the other 50% to accomplish the same task they have been doing before thus encouraging them to generalize on information they have neve seen before
        """
        cnn_model.add(Dropout(0.5))

        """
        dense to output reults for multi class classification - based on the 4 classes of tumor in BRISC2025
        activation softmax if the single label results - like only one true value in classification
        """
        cnn_model.add( Dense(len(CLASS_NAMES), activation="softmax")  )
        
        """
        Compile layer that finalazes the model 
        adam optimizer - turns loss gradients into weights updates, so basically the leanring step
        categorical crossentropy loss - penalizes the model when predicted probability is low and praises it when it is high 
        metrics for reporting
        accuracy - how many of my predictions are accurate - to assess true positives and true negatives
        precision - of the ones that the model predicted as truths , how many were truths  --- to assess false positives
        recall - of the ones that are actual truths how many did the model predict correctly  to assess false negatives
        """
        cnn_model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy", metrics.Precision(name="precision"), metrics.Recall(name="recall")]  )
        # from 86 
        
        print("Convolutional Neural Network model created")
        print("=" * 100)
            
        return cnn_model

    except:
        print("Error in develop cnn model function")
        return None


# function for training the model      
def train_cnn_model(augumented_data):
    try:     
        print("=" * 100)
        print("Training the CNN model...")
           
        # ensure the global cnn_model is being used in this function
        global cnn_model

        # collect necessary data for training purposes
        training_dataset = augumented_data["TRAINING_DATASET"]
        validation_dataset = augumented_data["VALIDATION_DATASET"]

		# a historical version of the trained model is to be stored
        trained_cnn_model_history = cnn_model.fit(training_dataset, validation_data=validation_dataset,  epochs=45, verbose=1)

		# save the current model under the file name
        cnn_model.save(MODEL_FILE)

		# from the historical version of the model display metrics tracked
        print("Epoch metrics tracked are: ",trained_cnn_model_history.history.keys())
        
        print("Training completed")
        print("=" * 100)
        
        return trained_cnn_model_history
        
    except:
        raise Exception ("Unable to train model")
        return None


# function to evaluate and predict
def evaluation_and_prediction(augumented_data):
     try:
        
        print("=" * 100)
        print("Evaluating and predicting on test data...")
        
        # access the global cnn_model variable
        global cnn_model 
        
        # get test data from augumented data
        testing_dataset = augumented_data["TESTING_DATASET"] 

        """
        evaluate testing dataset and print no ouputs with verbose 0
        loss - is always the first returned during evaluation - returns the loss averaged over each batch - how far off is the models prediction from actual truths. The lower the better
        accuracy, precison, recall - must come in that order as that is how they are created in the model
        """
        loss, accuracy, precision, recall = cnn_model.evaluate(testing_dataset, verbose=0) 

        # print evaluation metrics 
        print(f"Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall :  {recall}") 
        
        # model should perform prediction on test data - verbose 1 to show progress bar
        predictions = cnn_model.predict(testing_dataset, verbose=1) 
        
        # return evalations on test dat
        evaluated_data = { 
            "LOSS" :loss, 
            "ACCURACY" : accuracy, 
            "PRECISION" :precision, 
            "RECALL" : recall, 
            "PREDICTIONS" : predictions
        }
        
        print("Prediction task done")
        print("=" * 100)
        
        return evaluated_data 
        
     except:
        print("Unknown Error occured in evaluation and prediction function")
        return None


# function to evaluate the cnn model operformance
def model_performance_and_analysis(evaluated_data, augumented_data, data_preparation_details) : 
    try:
		# collect necessary data 
        predictions = evaluated_data["PREDICTIONS"]
        testing_dataset = augumented_data["TESTING_DATASET"]
        CLASS_NAMES = data_preparation_details["CLASS_NAMES"]
        
        """
        predictions are a basicaly a vector of probabilities, for each image using the one hot encodes a prediction looks like [0.1, 0.01, 0.88, 0] to say the probability is likely no_tumor
        so it looks like a numpy array(1000 images v 4 classes)
        my true predictions y_predictions will use numpy max function to assess and select the maximum values for each
        """
        y_predictions = np.argmax(predictions, axis=1)

        """
        testing dataset is similar to train, in that we know its image and labels
        for_, y in testing dataset - uisng _ means ignore the images sides reduce reading data not needed but keep the labels
        we convert the labels one hot encodes to singular values using the numpy maximum
        concatenate, will merge each in a long simple array - easy for classification report and confusion matrix
        """
        y_actual_truths = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in testing_dataset])

        # generate the confusion matrix
        the_confusion_matrix = confusion_matrix(y_actual_truths, y_predictions)

        # use ConfusionMatrixDisplay class to prepare the results for display
        display_confusion_matrix = ConfusionMatrixDisplay(the_confusion_matrix, display_labels=CLASS_NAMES)

        # plot on graph GUI - to improve user experience with more familiar parameters
        display_confusion_matrix.plot(cmap="Blues", values_format="d", xticks_rotation=45)
        plt.tight_layout()
        plt.show()
        
		# plot confusion matrix
        # print("The confusion matrix is as follows below: \n", the_confusion_matrix)

		# fetch classification report
        the_classification_report = classification_report(y_actual_truths, y_predictions, target_names=CLASS_NAMES, digits=4)
        
        # show on terminal only
        print("The classifcation report is as follows:   \n", the_classification_report)
         
    except:
        print("Error in evaluating model performance function")


# predict on a single image function 
def predict_on_single_image(augumented_data, data_preparation_details):
    try:
        # collect necessary data
        testing_dataset = augumented_data["TESTING_DATASET"] 
        CLASS_NAMES = data_preparation_details["CLASS_NAMES"] 
        BATCH_SIZE = data_preparation_details["BATCH_SIZE"] 
        
        # keep program in this loop unless user opts out
        while True: 
            # select a random batch using numpy random between 0 - first index of batch to BATCH SIZE = 32
            i = np.random.randint(0, BATCH_SIZE) 
            
            # fetch images and respective image labels from a random batch - from the ith batch
            for images, image_labels in testing_dataset.take(i): 
                random_index = np.random.randint(0, images.shape[0]) # random image index within the batch
                random_image = images[random_index].numpy() # get the random image from the batch and convert to numpy array
                true_onehot = image_labels[random_index].numpy() # get the true one-hot encoded label for the random image
                true_index = int(np.argmax(true_onehot)) # convert one-hot encoded label to class index
                true_name = CLASS_NAMES[true_index] # get the true class name from class index
                
            image_probability = cnn_model.predict(np.expand_dims(random_image, axis=0), verbose=0)[0] # predict class probabilities for the random image by expanding dimensions to match model input shape
            
            prediction_index = int(np.argmax(image_probability)) # get the predicted class index by taking the index of the max probability
            prediction_name = CLASS_NAMES[prediction_index] # get the predicted class name from class index
            confidence_score = float(image_probability[prediction_index]) # get the confidence score for the predicted class
            
            plt.imshow(random_image) # display the random image
            plt.axis("off") # turn off axis
            plt.title(f"Predicted Type: {prediction_name} || Confidence: {confidence_score:2%} || Actual Type: {true_name}") # set title with predicted class, confidence score, and true class name
            plt.show() # show the image

			# ask user if they wish to predict on a seperate test image
            option = input("Do you want to predict another random image (Say No to exit):   ").lower()

			# stop program only if user says no
            if option == "no":
                print("Thanks for using brAIniac")
                break
  
    except:
        print("unexpected error in predicting a single image function")
    
    

# main function to link all other functions together
def main():
    try:
        
        
        # call cumpulosry depenedent and helper functions
        prepared_data_details = data_preparation()
        preprocessed_data = data_loading_and_preprocessing(prepared_data_details)
        augmented_data = data_augmentation(preprocessed_data)
        
        # use global cnn model
        global cnn_model
        

  
        cnn_model = develop_cnn_model(prepared_data_details)
        history = train_cnn_model(augmented_data)
            

        evaluated_test_data = evaluation_and_prediction (augmented_data)
        model_performance_and_analysis(evaluated_test_data, augmented_data, prepared_data_details)
        predict_on_single_image(augmented_data, prepared_data_details)
        
    except:
        print("An error has occured in main function")
        

# run brAIniac
if __name__ == "__main__":
    main()  