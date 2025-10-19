# Imports Used

"""
OS - for file system operations and path checks on local machine
RANDOM - for reproducibility (randomness != reproducibility). reproducibility is to geneate same results repeatedly 
NUMPY - for array math
TENSORFLOW - backbone for keras and input pipelines without which keras can't optimize properly
MATPLOTLIB.PYLOT - To show images on a plot graph GUI
"""
import os, random, numpy as np, tensorflow as tf , matplotlib.pyplot as plt 


from keras.models import Sequential  
  

from keras.utils import image_dataset_from_directory

"""
LAYERS - to access layers.Rescalling class for image normalization and to access data augmentation layers
METRICS - to access metrics such as Precision and Recall
"""
from keras import layers, metrics


from keras.optimizers import Adam     


from keras.layers import Conv2D, MaxPooling2D , Flatten , Dense, Dropout


from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


cnn_model= Sequential()


SEED = 2025


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


AUTOTUNE = tf.data.AUTOTUNE


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

	
        IMAGE_SIZE = (224,224)

        BATCH_SIZE = 32

		
        CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

		
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

       
		# create training dataset by using the function image_dataset_from_directory
        training_dataset = image_dataset_from_directory (
            TRAIN_DIRECTORY,              
            labels="inferred",          	
            label_mode="categorical",      	
            color_mode="rgb",             
            image_size= IMAGE_SIZE,         
            batch_size=BATCH_SIZE,         	
            shuffle=True,                 	
            seed=SEED,                     	
            validation_split= 0.1,        
            subset="training"             	
        )            

        # create the valiation dataset - to act as a pop quiz after every epoch of training
        validation_dataset = image_dataset_from_directory (
            TRAIN_DIRECTORY, 
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb", 
            image_size= IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,                       
            seed=SEED,
            validation_split= 0.1,		
            subset="validation"          	           
        )

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


        rescale_pixels = layers.Rescaling(1./255)

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
     
                rescale_pixels(data_augmentation(x)),  y
            )    
        )

        # keep preprocessde data in ram for easy fetching and fine tunes performance using autotune

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


        global cnn_model

       
        CLASS_NAMES = data_preparation_details["CLASS_NAMES"]

		
        filters = 32

        cnn_model.add(Conv2D(filters, (3,3) , input_shape = (224,224,3), padding="same", activation = "relu"))
        
      
        cnn_model.add(MaxPooling2D(pool_size = (2, 2) ))
                
        # CNN BLOCK 2
       
        filters *= 2    
        cnn_model.add(Conv2D(filters, (3,3), padding="same", activation = "relu"))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2) ))
        
        # CNN BLOCK 3
        filters *= 2   
        cnn_model.add(Conv2D(filters, (3,3), padding="same", activation = "relu"))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2) ))

        cnn_model.add(Flatten())


        cnn_model.add( Dense (units = 128, activation = "relu") )

        cnn_model.add(Dropout(0.5))

   
        cnn_model.add( Dense(len(CLASS_NAMES), activation="softmax")  )
        

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
        trained_cnn_model_history = cnn_model.fit(training_dataset, validation_data=validation_dataset,  epochs=15, verbose=1)

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

        loss, accuracy, precision, recall = cnn_model.evaluate(testing_dataset, verbose=0) 

        # print evaluation metrics 
        print(f"Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall :  {recall}") 
        
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
		
        predictions = evaluated_data["PREDICTIONS"]
        testing_dataset = augumented_data["TESTING_DATASET"]
        CLASS_NAMES = data_preparation_details["CLASS_NAMES"]
        

        y_predictions = np.argmax(predictions, axis=1)


        y_actual_truths = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in testing_dataset])

      
        the_confusion_matrix = confusion_matrix(y_actual_truths, y_predictions)

      
        display_confusion_matrix = ConfusionMatrixDisplay(the_confusion_matrix, display_labels=CLASS_NAMES)

      
        display_confusion_matrix.plot(cmap="Blues", values_format="d", xticks_rotation=45)
        plt.tight_layout()
        plt.show()
        

        the_classification_report = classification_report(y_actual_truths, y_predictions, target_names=CLASS_NAMES, digits=4)
        
       
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
        
       
        while True: 
           
            i = np.random.randint(0, BATCH_SIZE) 
            
          
            for images, image_labels in testing_dataset.take(i): 
                random_index = np.random.randint(0, images.shape[0]) 
                random_image = images[random_index].numpy() 
                true_onehot = image_labels[random_index].numpy() 
                true_index = int(np.argmax(true_onehot))
                true_name = CLASS_NAMES[true_index] 
                
            image_probability = cnn_model.predict(np.expand_dims(random_image, axis=0), verbose=0)[0] 
            
            prediction_index = int(np.argmax(image_probability)) 
            prediction_name = CLASS_NAMES[prediction_index] 
            confidence_score = float(image_probability[prediction_index]) 
            
            plt.imshow(random_image) 
            plt.axis("off") # turn off axis
            plt.title(f"Predicted Type: {prediction_name} || Confidence: {confidence_score:2%} || Actual Type: {true_name}") 
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
