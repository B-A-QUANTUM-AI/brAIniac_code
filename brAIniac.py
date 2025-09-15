# leave the first 2 lines of code here 
from keras.models import Sequential
cnn_model= Sequential()




def data_preparation():
    try:
        #
        print("========================================================================================")
        print("Fetching and preparing the Data...")
        #
        
        import os
        
        BASE_DIRECTORY = "C:/Users/chidi/Downloads/brisc2025/classification_task"
        
        if not os.path.exists(BASE_DIRECTORY):
            raise FileNotFoundError("Classification folder was not found")
        
        TRAIN_DIRECTORY = os.path.join(BASE_DIRECTORY, "train")
        TEST_DIRECTORY = os.path.join(BASE_DIRECTORY, "test")
        
        if not os.path.exists(TRAIN_DIRECTORY):
            raise FileNotFoundError("Train folder was not found")
        
        if not os.path.exists(TEST_DIRECTORY):
            raise FileNotFoundError("Test folder was not found")
        
        IMAGE_SIZE = (224,224)
        BATCH_SIZE = 32
        SEED = 42
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
        
        #
        print("========================================================================================")
        print("Fetching and preparing task done")
        #
        
        return data_preparation_details
    
    except FileNotFoundError as e:
        print ("Error message:  ", e)
        return None
    except:
        print("An unexpected error during in data preparation")
        return None