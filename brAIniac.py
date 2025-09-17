# import sequential class and creat a sequential model to be acessed in necessary functions
from keras.models import Sequential
cnn_model= Sequential()
# tf for performance utilities
import os, random, numpy as np, tensorflow as tf

# global seed variable used to ensure consistent shuffle/splits per run
SEED = 2025

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# use tf autotune to fine tune performance on local machine
AUTOTUNE = tf.data.AUTOTUNE

def data_preparation():
    try:
        print("=" * 100)
        print("Fetching and preparing the Data...")
       
        import os
        
        BASE_DIRECTORY = "C:/Users/orera/Downloads/brisc2025/classification_task"
        
        if not os.path.exists(BASE_DIRECTORY):
            raise FileNotFoundError("Classification folder was not found")
        
        TRAIN_DIRECTORY = os.path.join(BASE_DIRECTORY, "train")
        TEST_DIRECTORY = os.path.join(BASE_DIRECTORY, "test")
        
        if not os.path.exists(TRAIN_DIRECTORY):
            raise FileNotFoundError("Train folder was not found")
        
        if not os.path.exists(TEST_DIRECTORY):
            raise FileNotFoundError("Test folder was not found")
        
        IMAGE_SIZE = (224,224)
        
        # number of images per training babatch
        BATCH_SIZE = 32
     
        CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
        
      #  WEIGHT_COUNTS = {cls: len(os.listdir(os.path.join(TRAIN_DIRECTORY, cls))) for cls in CLASS_NAMES}
        
        data_preparation_details = {
            "BASE_DIRECTORY" : BASE_DIRECTORY,
            "TRAIN_DIRECTORY" : TRAIN_DIRECTORY,
            "TEST_DIRECTORY": TEST_DIRECTORY,
            "IMAGE_SIZE" : IMAGE_SIZE,
            "BATCH_SIZE" : BATCH_SIZE,
            "SEED" : SEED,
            "CLASS_NAMES" : CLASS_NAMES,
           # "WEIGHT_COUNTS" : WEIGHT_COUNTS
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
    
def data_loading_and_preprocessing(prepared_data):
    
    try:
        print("=" * 100)
        print("Loading Data ...")
        
	#Loads Keras function to read images directly from the image dataset folder
        from keras.utils import image_dataset_from_directory
        
        TRAIN_DIRECTORY = prepared_data["TRAIN_DIRECTORY"]
        TEST_DIRECTORY = prepared_data["TEST_DIRECTORY"]
        IMAGE_SIZE = prepared_data["IMAGE_SIZE"]
        BATCH_SIZE = prepared_data["BATCH_SIZE"]
        SEED = prepared_data["SEED"]
        

        training_dataset = image_dataset_from_directory (
            TRAIN_DIRECTORY,               
            labels="inferred",          
            label_mode="categorical",      
            color_mode="rgb",               
            image_size= IMAGE_SIZE,         
            batch_size=BATCH_SIZE,         
            shuffle=True,                 # randomize image order
            seed=SEED,                     
            validation_split= 0.1,        # 10% of the data is reserved for validation  
            subset="training"             
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
            subset="validation"                    
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
    
    
    
def data_augmentation(preprocessed_data):
    
    try:       
        print("=" * 100)
        print("Augmenting imaes")

        training_dataset = preprocessed_data["TRAINING_DATASET"]
        validation_dataset = preprocessed_data["VALIDATION_DATASET"]
        testing_dataset = preprocessed_data["TESTING_DATASET"]
    
        from keras import layers
            
        rescale_pixels = layers.Rescaling(1./255)
                
        from keras.models import Sequential
             
        data_augmentation = Sequential (
            [
                layers.RandomContrast(0.1),       
                layers.RandomRotation(0.05),            
                layers.RandomZoom(0.10),               
                layers.RandomTranslation(0.05, 0.05)    
            ],
            name= "augment"                             
        )
               
        training_dataset = training_dataset.map(
            lambda x, y :    (
                # applying augmentation to data
                # applied normalization and augmentation to images in this dataset || y is the label so remains same is the name of the image 
                rescale_pixels(data_augmentation(x)),  y
            )    
        )

        # keep preprocessde data in ram for easy fetching and fine tunes performance using autotune
        training_dataset = training_dataset.cache().prefetch(AUTOTUNE)       

        validation_dataset = validation_dataset.map(
            lambda x, y : (
                    rescale_pixels(x)  , y 
            ) 
        )

        validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)   
       
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