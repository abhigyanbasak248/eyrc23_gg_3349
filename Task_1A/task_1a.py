'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas 
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################





##############################################################

def data_preprocessing(task_1a_dataframe):

    ''' 
    Purpose:
    ---
    This function will be used to load your csv dataset and preprocess it.
    Preprocessing involves cleaning the dataset by removing unwanted features,
    decision about what needs to be done with missing values etc. Note that 
    there are features in the csv file whose values are textual (eg: Industry, 
    Education Level etc)These features might be required for training the model
    but can not be given directly as strings for training. Hence this function 
    should return encoded dataframe in which all the textual features are 
    numerically labeled.
    
    Input Arguments:
    ---
    `task_1a_dataframe`: [Dataframe]
                          Pandas dataframe read from the provided dataset 	
    
    Returns:
    ---
    `encoded_dataframe` : [ Dataframe ]
                          Pandas dataframe that has all the features mapped to 
                          numbers starting from zero

    Example call:
    ---
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    '''
    le = LabelEncoder()
    task_1a_dataframe['Education'] = le.fit_transform(task_1a_dataframe['Education'])
    task_1a_dataframe['JoiningYear'] = le.fit_transform(task_1a_dataframe['JoiningYear'])
    task_1a_dataframe['City'] = le.fit_transform(task_1a_dataframe['City'])
    task_1a_dataframe['Gender'] = le.fit_transform(task_1a_dataframe['Gender'])
    task_1a_dataframe['EverBenched'] = le.fit_transform(task_1a_dataframe['EverBenched'])
    # df_majority = task_1a_dataframe[task_1a_dataframe.LeaveOrNot==0]
    # df_minority = task_1a_dataframe[task_1a_dataframe.LeaveOrNot==1]
 
    # df_minority_upsampled = resample(df_minority, 
    #                              replace=True,
    #                              n_samples=3041,
    #                              random_state=123)
    # df_upsampled = pandas.concat([df_majority, df_minority_upsampled])
    # encoded_dataframe = df_upsampled
    task_1a_dataframe = task_1a_dataframe.drop(['ExperienceInCurrentDomain'], axis = 1)
    encoded_dataframe = task_1a_dataframe
    return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
    '''
    Purpose:
    ---
    The purpose of this function is to define the features and
    the required target labels. The function returns a python list
    in which the first item is the selected features and second 
    item is the target label

    Input Arguments:
    ---
    `encoded_dataframe` : [ Dataframe ]
                        Pandas dataframe that has all the features mapped to 
                        numbers starting from zero
    
    Returns:
    ---
    `features_and_targets` : [ list ]
                            python list in which the first item is the 
                            selected features and second item is the target label

    Example call:
    ---
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    '''
    X = encoded_dataframe.iloc[:, :8]
    # X = X / (X.max() - X.min())
    X = X - X.mean() / X.std()
    y = encoded_dataframe['LeaveOrNot']
    features_and_targets = [X, y]

    return features_and_targets


def load_as_tensors(features_and_targets):

    ''' 
    Purpose:
    ---
    This function aims at loading your data (both training and validation)
    as PyTorch tensors. Here you will have to split the dataset for training 
    and validation, and then load them as as tensors. 
    Training of the model requires iterating over the training tensors. 
    Hence the training sensors need to be converted to iterable dataset
    object.
    
    Input Arguments:
    ---
    `features_and targets` : [ list ]
                            python list in which the first item is the 
                            selected features and second item is the target label
    
    Returns:
    ---
    `tensors_and_iterable_training_data` : [ list ]
                                            Items:
                                            [0]: X_train_tensor: Training features loaded into Pytorch array
                                            [1]: X_test_tensor: Feature tensors in validation data
                                            [2]: y_train_tensor: Training labels as Pytorch tensor
                                            [3]: y_test_tensor: Target labels as tensor in validation data
                                            [4]: Iterable dataset object and iterating over it in 
                                                 batches, which are then fed into the model for processing

    Example call:
    ---
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    '''
    X_train, X_test, y_train, y_test = train_test_split(features_and_targets[0], features_and_targets[1], test_size = 0.2, random_state = 42)
    X_train_tensor = torch.tensor(X_train.values, dtype = torch.float32)
    # X_train_tensor = X_train_tensor[None, ...]
    X_test_tensor = torch.tensor(X_test.values, dtype = torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype = torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype = torch.float32)
    data = TensorDataset(torch.tensor(features_and_targets[0].values, dtype = torch.float32), torch.tensor(features_and_targets[1].values, dtype = torch.float32))
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    tensors_and_iterable_training_data = [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, dataloader]

    return tensors_and_iterable_training_data

class Salary_Predictor(nn.Module):
    '''
    Purpose:
    ---
    The architecture and behavior of your neural network model will be
    defined within this class that inherits from nn.Module. Here you
    also need to specify how the input data is processed through the layers. 
    It defines the sequence of operations that transform the input data into 
    the predicted output. When an instance of this class is created and data
    is passed through it, the `forward` method is automatically called, and 
    the output is the prediction of the model based on the input data.
    
    Returns:
    ---
    `predicted_output` : Predicted output for the given input data
    '''
    def __init__(self):
        super(Salary_Predictor, self).__init__()
        '''
        Define the type and number of layers
        '''
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(8, 32)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        self.BN1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        self.BN2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.BN3= nn.BatchNorm1d(64)
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc3.bias)
        self.fc4 = nn.Linear(64, 1)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
# =============================================================================
#         self.layers = nn.Sequential(
#             nn.Linear(8, 16),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.Linear(16,8),
#             nn.BatchNorm1d(8),
#             nn.ReLU(),
#             nn.Linear(8, 1),
#             nn.Sigmoid())
# =============================================================================

    def forward(self, x):
        '''
        Define the activation functions
        '''
        try:
            x.shape[1]
        except:
            x = x[None, ...]
            
        x = self.fc1(x)
        # x = x.unsqueeze(1)
        # x = (x - x.mean())/x.std()
        x = F.relu(self.BN1(x))
        # x = self.BN2(F.relu(self.fc2(x)))
        x = self.fc2(x)
        # x = x.unsqueeze(1)
        # x = (x - x.mean())/x.std()
        x = F.relu(self.BN2(x))
        x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)
        x = F.relu(self.BN3(x))
        x = self.dropout(x)
        predicted_output = F.sigmoid(self.fc4(x))
        return predicted_output

def model_loss_function():
    '''
    Purpose:
    ---
    To define the loss function for the model. Loss function measures 
    how well the predictions of a model match the actual target values 
    in training data.
    
    Input Arguments:
    ---
    None

    Returns:
    ---
    `loss_function`: This can be a pre-defined loss function in PyTorch
                    or can be user-defined

    Example call:
    ---
    loss_function = model_loss_function()
    '''
    loss_function = nn.BCELoss()
    
    return loss_function

def model_optimizer(model):
    '''
    Purpose:
    ---
    To define the optimizer for the model. Optimizer is responsible 
    for updating the parameters (weights and biases) in a way that 
    minimizes the loss function.
    
    Input Arguments:
    ---
    `model`: An object of the 'Salary_Predictor' class

    Returns:
    ---
    `optimizer`: Pre-defined optimizer from Pytorch

    Example call:
    ---
    optimizer = model_optimizer(model)
    '''
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    return optimizer

def model_number_of_epochs():
    '''
    Purpose:
    ---
    To define the number of epochs for training the model

    Input Arguments:
    ---
    None

    Returns:
    ---
    `number_of_epochs`: [integer value]

    Example call:
    ---
    number_of_epochs = model_number_of_epochs()
    '''
    number_of_epochs = 100

    return number_of_epochs

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    '''
    Purpose:
    ---
    All the required parameters for training are passed to this function.

    Input Arguments:
    ---
    1. `model`: An object of the 'Salary_Predictor' class
    2. `number_of_epochs`: For training the model
    3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and iterable dataset object of training tensors
    4. `loss_function`: Loss function defined for the model
    5. `optimizer`: Optimizer defined for the model

    Returns:
    ---
    trained_model

    Example call:
    ---
    trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

    '''	
    # batch_size = 10  # size of each batch
    # batch_start = torch.arange(0, len(tensors_and_iterable_training_data[0]), batch_size)
    model.train()
    for epoch in range(number_of_epochs):
                # X_batch = tensors_and_iterable_training_data[0][start:start+batch_size]
                # y_batch = tensors_and_iterable_training_data[2][start:start+batch_size]
        #forward
        y_pred = model(tensors_and_iterable_training_data[0])
        loss = loss_function(y_pred, tensors_and_iterable_training_data[2].reshape(-1, 1))
    
        optimizer.zero_grad()

        #backward
        loss.backward()
        optimizer.step()
        
        # Print the loss for this epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}], Loss: {loss.item():.4f}")
    trained_model = model

    return trained_model

# =============================================================================
# def model_train(model, X_train, y_train, X_val, y_val):
#     # loss function and optimizer
#     loss_fn = nn.BCELoss()  # binary cross entropy
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#   
#     n_epochs = 300   # number of epochs to run
#     batch_size = 10  # size of each batch
#     batch_start = torch.arange(0, len(X_train), batch_size)
#   
#     # Hold the best model
#     best_acc = - np.inf   # init to negative infinity
#     best_weights = None
#   
#     for epoch in range(n_epochs):
#         model.train()
#         with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
#             bar.set_description(f"Epoch {epoch}")
#             for start in bar:
#                 # take a batch
#                 X_batch = X_train[start:start+batch_size]
#                 y_batch = y_train[start:start+batch_size]
#                 # forward pass
#                 y_pred = model(X_batch)
#                 loss = loss_fn(y_pred, y_batch.reshape(-1, 1))
#                 # backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 # update weights
#                 optimizer.step()
#                 # print progress
#                 acc = (y_pred.round() == y_batch).float().mean()
#                 bar.set_postfix(
#                     loss=float(loss),
#                     acc=float(acc)
#                 )
#         
#         model.eval()
#         y_pred = model(X_val)
#         correct = 0
#         for i in range(len(y_pred)):
#             if (y_pred[i] >= 0.5):
#                  y_pred[i] = 1.
#             else:
#                 y_pred[i] = 0.
#         ind = 0
#         for i in y_val:
#             if (i == y_pred[ind]):
#                   correct += 1
#             ind += 1
#             acc = correct / y_val.size(0)
#         acc = 100 * acc
#         # acc = (y_pred.round() == y_val).float().mean()
#         # acc = float(acc)
#         if acc > best_acc:
#             best_acc = acc
#             # best_weights = copy.deepcopy(model.state_dict())
#     # restore model and return best accuracy
#     # model.load_state_dict(best_weights)
#     return best_acc
# =============================================================================

def validation_function(trained_model, tensors_and_iterable_training_data):
    '''
    Purpose:
    ---
    This function will utilise the trained model to do predictions on the
    validation dataset. This will enable us to understand the accuracy of
    the model.

    Input Arguments:
    ---
    1. `trained_model`: Returned from the training function
    2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
                                             and iterable dataset object of training tensors

    Returns:
    ---
    model_accuracy: Accuracy on the validation dataset

    Example call:
    ---
    model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

    '''	
    trained_model.eval()
    correct = 0
    y_pred = trained_model(tensors_and_iterable_training_data[1])
    for i in range(len(y_pred)):
        if (y_pred[i] >= 0.5):
            y_pred[i] = 1.
        else:
              y_pred[i] = 0
             
    ind = 0
    for i in tensors_and_iterable_training_data[3]:
        if (i == y_pred[ind]):
              correct += 1
        ind += 1
    acc = correct / tensors_and_iterable_training_data[3].size(0)
    model_accuracy = 100 * acc

# =============================================================================
#     kfold = StratifiedKFold(n_splits=5, shuffle=True)
#     cv_scores = []
#     for train, test in kfold.split(tensors_and_iterable_training_data[0], tensors_and_iterable_training_data[2]):
#         # create model, train, and get accuracy
#         model = Salary_Predictor()
#         acc = model_train(model, tensors_and_iterable_training_data[0], tensors_and_iterable_training_data[2], tensors_and_iterable_training_data[1], tensors_and_iterable_training_data[3])
#         print("Accuracy (wide): %.2f" % acc)
#         cv_scores.append(acc)
#  
#     # evaluate the model
#     model_accuracy = np.mean(cv_scores)
#     std = np.std(cv_scores)
#     print("Model accuracy: %.2f%% (+/- %.2f%%)" % (model_accuracy, std*100))
# =============================================================================
    return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
    Purpose:
    ---
    The following is the main function combining all the functions
    mentioned above. Go through this function to understand the flow
    of the script
'''
if __name__ == "__main__":

    # reading the provided dataset csv file using pandas library and 
    # converting it to a pandas Dataframe
    task_1a_dataframe = pandas.read_csv('task_1a_dataset.csv')

    # data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)

    # selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)

    # obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
    
    # model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor()

    # obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()

    # training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
                    loss_function, optimizer)

    # validating and obtaining accuracy
    model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")

    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")