# imports the sequential class to build neural networks layer by layer
from keras.models import Sequential

# create the sequential cnn model
cnn_model= Sequential()

# tf for performance utilities
import os, random, numpy as np, tensorflow as tf

# global seed variable used to ensure consistent shuffle/splits per run
SEED = 2025

# randomize seeds using python's in built random, numpy's random and tensorflow's random
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# constant telling tensor flow to automatically optimize data loading performance
AUTOTUNE = tf.data.AUTOTUNE

# data preparation function
def data_preparation():
    try:
        print("=" * 100)
        print("Fetching and preparing the Data...")
       
        import os
		# change name to your pc name to match location of your dataset directory on local computer
        computer_name = "orera"
        BASE_DIRECTORY = f"C:/Users/{computer_name}/Downloads/brisc2025/classification_task"
        
        if not os.path.exists(BASE_DIRECTORY):
            raise FileNotFoundError("Classification folder was not found")
        
        TRAIN_DIRECTORY = os.path.join(BASE_DIRECTORY, "train")
        TEST_DIRECTORY = os.path.join(BASE_DIRECTORY, "test")
        
        if not os.path.exists(TRAIN_DIRECTORY):
            raise FileNotFoundError("Train folder was not found")
        
        if not os.path.exists(TEST_DIRECTORY):
            raise FileNotFoundError("Test folder was not found")

		# set image size for every iamge in dataset
        IMAGE_SIZE = (224,224)

		# no of images to process at a time
        BATCH_SIZE = 32

		# names of tumors according to the folders alphabetical order
        CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

		# information to be returned from the function
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
        print("An unexpected error during in data preparation")
        return None

# data loading function
def data_loading_and_preprocessing(prepared_data):
    
    try:
        print("=" * 100)
        print("Loading Data ...")
        
		#Loads Keras function to read images directly from the image dataset folder
        from keras.utils import image_dataset_from_directory

		# collect neccessary info from previous function
        TRAIN_DIRECTORY = prepared_data["TRAIN_DIRECTORY"]
        TEST_DIRECTORY = prepared_data["TEST_DIRECTORY"]
        IMAGE_SIZE = prepared_data["IMAGE_SIZE"]
        BATCH_SIZE = prepared_data["BATCH_SIZE"]
        SEED = prepared_data["SEED"]

		# create data set by using the function image dataset from directory in the Keras.utils 
        training_dataset = image_dataset_from_directory (
            TRAIN_DIRECTORY,               	# directory to train data
            labels="inferred",          	# infer the labels and truths from the class folders
            label_mode="categorical",      	# to create one - hot labels styles such as [0,1,0,0] - representing an image is from meningioma class
            color_mode="rgb",               # set the color format
            image_size= IMAGE_SIZE,         # resize all images
            batch_size=BATCH_SIZE,         	# no of images per batch
            shuffle=True,                 	# randomize image order
            seed=SEED,                     	# make the shuffle reproducable
            validation_split= 0.1,        	# reserve 10% of the data is reserved for validation  
            subset="training"             	# theis current dataset is for training not the validation
        )            

        validation_dataset = image_dataset_from_directory (
            TRAIN_DIRECTORY, 
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb", 
            image_size= IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,                   # for steady validation batches every run      
            seed=SEED,
            validation_split= 0.1,		
            subset="validation"          	 # this subset is for validation so the reserved 10 %           
        )

        testing_dataset = image_dataset_from_directory (
            TEST_DIRECTORY, 
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb", 
            image_size= IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False                   
        )

        print("Class order:", training_dataset.class_names)        
        
        preprocessed_data = {
            "TRAINING_DATASET" : training_dataset,
            "VALIDATION_DATASET" : validation_dataset,
            "TESTING_DATASET" : testing_dataset
        }
        
        print("Data has been loaded")
        print("=" * 100)

        return preprocessed_data
    
    except:
        print("Unexpected error in Data loading function")
        return None
    
# data augmentation function for augmenting the images
def data_augmentation(preprocessed_data):
    
    try:       
        print("=" * 100)
        print("Augmenting imaes")

		# collect neccessary info from previous function
        training_dataset = preprocessed_data["TRAINING_DATASET"]
        validation_dataset = preprocessed_data["VALIDATION_DATASET"]
        testing_dataset = preprocessed_data["TESTING_DATASET"]
    
        from keras import layers

		# normalization algorithm for images
        rescale_pixels = layers.Rescaling(1./255)
                
        from keras.models import Sequential

		# create a sequential model to create an augmentation alogrithm
        data_augmentation = Sequential (
            [
                layers.RandomContrast(0.1),       
                layers.RandomRotation(0.05),            
                layers.RandomZoom(0.10),               
                layers.RandomTranslation(0.05, 0.05)    
            ],
            name= "augment"                             
        )

		# apply the augmentation and normalization to training dataset
        training_dataset = training_dataset.map(
            lambda x, y :    (
                # applying augmentation to data
                # applied normalization and augmentation to images in this dataset || y is the label so remains same is the name of the image 
                rescale_pixels(data_augmentation(x)),  y
            )    
        )

        # keep preprocessde data in ram for easy fetching and fine tunes performance using autotune
        training_dataset = training_dataset.cache().prefetch(AUTOTUNE)       

		# just normalize images
        validation_dataset = validation_dataset.map(
            lambda x, y : (
                    rescale_pixels(x)  , y 
            ) 
        )

        validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)   

		# normalize test data
        testing_dataset = testing_dataset.map(
            lambda x, y : (
                rescale_pixels(x)  , y 
            ) 
        )

        testing_dataset = testing_dataset.cache().prefetch(AUTOTUNE)      
        
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
    
# developing the sequentioal model by layers
def develop_cnn_model(prepared_data):
    
    try :

        print("=" * 100)
        print("Developing the Convolutional Neural Network model...")

        global cnn_model
        
        from keras.layers import Conv2D, MaxPooling2D , Flatten , Dense, Dropout

        CLASS_NAMES = prepared_data["CLASS_NAMES"]

		# number of filters needed to learn from each image
        filters = 32

		# loop 3 times to create 3 blocks of convolution and pooling layers
        for i in range(3):
        
            cnn_model.add(
				# examines each image and tries to learn features from each image
                Conv2D(
                    # activation - relu - to allow model learn more complex mpatterns || 
					# (3,3) is the size of each filter 
                    filters, (3,3) , input_shape = (224,224,3), activation = "relu" 
                )
            )

            cnn_model.add(
				# collects all learned features together
                MaxPooling2D (
                    # reduce spatial dimension of images by half on width and height
                    pool_size = (2, 2)
                )   
            )

			# increase no  of filters by 2 
            filters *= 2
            
                
        # flatten to 1D vector for easy use by dense layer
        cnn_model.add(Flatten())

        # dense layer with 128 neurons to learn high combo features sing activation relu
        cnn_model.add( Dense (units = 128, activation = "relu")   )

        # drop 50% units to avoidoverfitting
        cnn_model.add(Dropout(0.5))

        # dense to output for each of the 4 classes
        cnn_model.add( Dense(len(CLASS_NAMES), activation="softmax")  )
            
        from keras import metrics
        from keras.optimizers import Adam

        #adam - adaptive optimization algorithm
		# indicate performance metrics to track
        cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", metrics.Precision(name="precision"), metrics.Recall(name="recall")]  )


        print("Convolutional Neural Network model created")
        print("=" * 100)
            
        return cnn_model

    except:
        print("Error in developing cn model ")
        return None

    
# function for training the model      
def train_cnn_model(augumented_data):
    
    try:     
        print("=" * 100)
        print("Training the CNN model...")
           
        global cnn_model

		# store the file name of a previous model save
        MODEL_FILE = "brainiac_cnn_86.keras"
        
        training_dataset = augumented_data["TRAINING_DATASET"]
        validation_dataset = augumented_data["VALIDATION_DATASET"]

        from keras.models import Sequential

		# a historical version of the trained model is to be stored
        trained_cnn_model_history = cnn_model.fit(training_dataset, validation_data=validation_dataset,  epochs=20, verbose=1)

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
def evaluation_and_prediction(augumented_data): # evaluate and predict on test data using augumented data
    
     try:
        
        print("=" * 100)
        print("Evaluating and predicting on test data...")
        
        global cnn_model # access the global cnn_model variable
        
        testing_dataset = augumented_data["TESTING_DATASET"] # get test data from augumented data
        
        import matplotlib.pyplot as plt
        
        import numpy as np

        loss, accuracy, precision, recall = cnn_model.evaluate(testing_dataset, verbose=0) # verbose 0 to avoid too much output
        # Loss : Measures how far off the model's predictions are from the actual labels. Lower is better.
        # Accuracy : Indicates the proportion of correct predictions made by the model. Higher is better.
        # Precision : Measures the accuracy of positive predictions. Higher is better.  
        # Recall : Measures the model's ability to find all relevant cases (true positives). Higher is better.
        # verbose = 0 to avoid too much output
        # This method tests the model on new, unseen data (in this case, testing_dataset) and returns the loss,accuracy, precision, and recall metrics.

        print(f"Accuracy: {accuracy}, Precision(no of correct predictions): {precision}, recall :  {recall}") # print evaluation metrics 
        
        predictions = cnn_model.predict(testing_dataset, verbose=1) # verbose 1 to show progress bar
        # This method generates output predictions for the input samples in testing_dataset. The output is typically a probability distribution over the classes for each input sample.
        # verbose = 1 to show progress bar
        
        evaluated_data = { # store evaluation metrics and predictions in a dictionary
            "LOSS" :loss, 
            "ACCURACY" : accuracy, 
            "PRECISION" :precision, 
            "RECALL" : recall, 
            "PREDICTIONS" : predictions
        }
        
        print("Prediction task done")
        print("=" * 100)
        
        return evaluated_data # return the dictionary
        
     except:
        print("Unknown Error occured in evaluation and prediction function")
        return None


# function to evaluate the cnn model operformance
def model_performance_and_analysis(evaluated_data, augumented_data, data_preparation_details) : 
    try:
		# collect necessaryu infor fromprevious functions
        predictions = evaluated_data["PREDICTIONS"]
        testing_dataset = augumented_data["TESTING_DATASET"]
        CLASS_NAMES = data_preparation_details["CLASS_NAMES"]
        
        import numpy as np
        
        from sklearn.metrics import confusion_matrix, classification_report

		# converts predictions to point numbers 
        y_predictions = np.argmax(predictions, axis=1)

		# buil an array of truth labels, for loop to go through dataset by batches and numpy to convert to an array
        y_actual_truths = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in testing_dataset])

		# plot confusion matrix
        the_confusion_matrix = confusion_matrix(y_actual_truths, y_predictions)
        
        print("The confusion matrix is as follows below: \n", the_confusion_matrix)

		# fetch classification report
        the_classification_report = classification_report(y_actual_truths, y_predictions, target_names=CLASS_NAMES, digits=4)
        
        print("The classifcation report is as follows:   \n", the_classification_report)
         
    except:
        print("Error in evaluating model performance")


# predict on a single image function 
def predict_on_single_image(augumented_data, data_preparation_details):
    try:
		
        testing_dataset = augumented_data["TESTING_DATASET"] # testing dataset
        CLASS_NAMES = data_preparation_details["CLASS_NAMES"] # class names in the dataset
        BATCH_SIZE = data_preparation_details["BATCH_SIZE"] # batch size used during data loading
        
        import numpy as np
        
        while True : 
            i = np.random.randint(0, BATCH_SIZE) # random batch index
            
            for images, image_label in testing_dataset.take(i): # take the ith batch from the test dataset
            
                random_index = np.random.randint(0, images.shape[0]) # random image index within the batch
                random_image = images[random_index].numpy() # get the random image from the batch and convert to numpy array
                true_onehot = image_label[random_index].numpy() # get the true one-hot encoded label for the random image
                true_index = int(np.argmax(true_onehot)) # convert one-hot encoded label to class index
                true_name = CLASS_NAMES[true_index] # get the true class name from class index
                
            image_probability = cnn_model.predict(np.expand_dims(random_image, axis=0), verbose=0)[0] # predict class probabilities for the random image by expanding dimensions to match model input shape
            
            prediction_index = int(np.argmax(image_probability)) # get the predicted class index by taking the index of the max probability
            prediction_name = CLASS_NAMES[prediction_index] # get the predicted class name from class index
            confidence_score = float(image_probability[prediction_index]) # get the confidence score for the predicted class
            
            import matplotlib.pyplot as plt
            
            plt.imshow(random_image) # display the random image
            plt.axis("off") # turn off axis
            plt.title(f"Predicted: {prediction_name} || Confidence: {confidence_score:1%} || Actual Class: {true_name}") # set title with predicted class, confidence score, and true class name
            plt.show() # show the image

			# ask user if they wish to predict on a seperate test image
            option = input("Do you want to predict another random image (Say No to exit):   ").lower()

			# stop program only if user says no
            if option == "no":
                print("Thanks for using brainiac")
                break
  
    except:
        print("unexpected error in predicting a single image function")
    


def main():
    try:
        
        import os
        
        MODEL_FILE = "brainiac_cnn_86.keras"
        
        # call depenedent and helper functions
        prepared_data_details = data_preparation()
        preprocessed_data = data_loading_and_preprocessing(prepared_data_details)
        augmented_data = data_augmentation(preprocessed_data)
        
        global cnn_model
        
        if os.path.exists(MODEL_FILE):
            print("=" * 100)
            print(f"Loading exiting model from {MODEL_FILE}")
            
            from keras.models import load_model

            cnn_model = load_model(MODEL_FILE)
            
            print(f"Model has been loaded from {MODEL_FILE}")
        
        else:
  
            cnn_model = develop_cnn_model(prepared_data_details)
            history = train_cnn_model(augmented_data)
            
            
        evaluated_test_data = evaluation_and_prediction (augmented_data)
        model_performance_and_analysis(evaluated_test_data, augmented_data, prepared_data_details)
        predict_on_single_image(augmented_data, prepared_data_details)
        
    except:
        print("An error has occured in main function")
        


if __name__ == "__main__":
    main()
