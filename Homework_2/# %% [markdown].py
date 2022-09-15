# %% [markdown]
# # Assignment 2

# %% [markdown]
# Create a program to evaluate the Generalization Error (GE), Prediction Model Error (ME) and Training Error (TE) for the k-nearest neighbors (KNN) learning approach. For doing so, compute the model considering neighborhood sizes from 1 to 35.

# %% [markdown]
# ### Imports:

# %%
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# %% [markdown]
# ### Built-in Custom Functions:

# %%
def KNN(N_size, x_train, Y_train, x_test):
    '''
    This function implements the KNN regression learning model. The required
    inputs are the following:
    -   N_size (integer): size of the neighborhood. Automatically reduced to
    training dataset size if greater than it.
    -   x_train (1-D list): list of x values related to the training data set.
    -   Y_train (1-D list): list of Y values related to the training data set.
    -   x_test (1-D list): list of x values related to the testing data set.

    The function outputs the following:
    -   Y_hat (1-D list): list containing the KNN regressed values for the 
    x_test data set according to the model training.
    '''

    N_size = np.minimum(len(x_train), N_size)

    x_i = [[x] for x in x_train]

    KNN = KNeighborsRegressor(N_size).fit(x_i, Y_train)

    x_i = [[x] for x in x_test]

    Y_hat = KNN.predict(x_i)

    return Y_hat


# %% [markdown]
# ## Solution Code:

# %% [markdown]
# ### Data Sets and Learning Model:
# 

# %% [markdown]
# #### Training Set:
# 
# With $N^{training} = 50$:
# 
# 
# - Generate $x_i$, $N^{training}$ uniformly separated data points between 0 and 1.
# 
# - Generate $n_i$, $N^{training}$ noise data points randomly distributed with 0 mean and 0.1 variance.
# 
# - Build the observed data model as: 
# 
# $Y_i^{training} = f(x_i) + n_i$, with $\space i = 1 ... N^{training}$ and $f(x) = sin(2 \pi · x)$

# %%
# Defining function to generate a set of data
def gen_data(n_samples: int) -> np.ndarray:
    def func_x(input):
        return np.sin((2)*(np.pi)*(input))

    #x_i = np.linspace(0, 1, num=n_samples)

    x_i = []
    delx = 1/n_samples

    for x in range(n_samples):
        x_i.append(x*delx)
    

    #noise 
    n_i = np.random.normal(loc=0, scale=np.sqrt(0.1), size=n_samples)

    #combining features + noise
    generated_samples = func_x(np.array(x_i)) + n_i


    #Dataset size
    print(f'Shape of generated labels: {np.shape(generated_samples)}\nShape of generated inputs: {np.shape(np.array(x_i))}')

    return np.array(x_i), generated_samples 

# %% [markdown]
# ### Generating training set (`n_samples` = 50)

# %%
X_train, y_train = gen_data(n_samples=50)

# %% [markdown]
# #### Testing Set:
# 
# With $N^{testing} = 300$:
# 
# 
# - Follow the same previous steps, using $N^{testing}$ instead of $N^{training}$.

# %%
X_test, y_test = gen_data(n_samples=300)

# %%
print(f'y_train: {y_train}')
print('--------------------')
print(f'y_test: {y_test}')

# %% [markdown]
# #### Learning Model:
# 
# Use the K-Nearest Neighbors to evaluate its performance. Plot the model result for neighborhood sizes of 1, 5, 15, 25 and 40.

# %%
import matplotlib.pyplot as plt

# %%
N_sizes_plot = [1,2,5,35] #Plot these neighborhood sizes

legend_names=['K=1', 'K=2', 'K=5', 'K=35', 'Y_train']
for n in N_sizes_plot:
    y_hat = KNN(N_size=n, x_train=X_train, Y_train=y_train, x_test=X_test)
    plt.plot(X_test, y_hat)
    plt.legend(legend_names)

#plt.plot(X_train, y_train)
plt.title('Y_hat trained w/ KNN with differing neighborhood sizes')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()


# %%
plt.plot(np.sin((2)*(np.pi)*(X_test)))
plt.title('Actual Function: sin(2*pi*x)')
plt.ylabel('Y')
plt.xlabel('X')

# %%
plt.plot(X_train, y_train)
plt.plot(X_test, y_test, alpha=0.5)
plt.legend(['Training model', 'Testing model'])
plt.title('Training and Testing model based on f(x)')
plt.xlabel('X')
plt.ylabel('Y')

# %%
def plot_model_comparison(y_hat, k):
    plt.plot(X_test, np.sin(2*np.pi*X_test))
    plt.plot(X_test, y_hat)
    plt.plot(X_test, y_test, alpha=0.45)
    plt.title(f'Learning Model Comparison, K = {k}')
    plt.legend(['f(x)', 'KNN', 'y test'])
    plt.xlabel('X')
    plt.ylabel('Y')
    
for n in N_sizes_plot:
    y_hat = KNN(N_size=n, x_train=X_train, Y_train=y_train, x_test=X_test)
    plot_model_comparison(y_hat, k=n)
    plt.show()

# %% [markdown]
# Clearly, K = 5 fits it the best

# %% [markdown]
# ### Evaluation:

# %% [markdown]
# #### Error analysis

# %% [markdown]
# Note that I had to edit the KNN function provided above in order for the error analysis to work properly. There was currently an error where the previously provided KNN function fell short in calculating the training error because after regressing it would just predict on the testing set. However in the case of calculating TE we need the KNN to predict over the training set and that y_hat was used in the calculation of TE. I simply added an input argument `predict_test` which when `False` would use the training set to make y_hat predictions.

# %%
def KNN_edited(N_size, x_train, Y_train, x_test, predict_test:bool):
    '''
    This function implements the KNN regression learning model. The required
    inputs are the following:
    -   N_size (integer): size of the neighborhood. Automatically reduced to
    training dataset size if greater than it.
    -   x_train (1-D list): list of x values related to the training data set.
    -   Y_train (1-D list): list of Y values related to the training data set.
    -   x_test (1-D list): list of x values related to the testing data set.
    -   predict_test (boolean): True or False wether or not you want to make predictions on the testing set. 
    If False it will default and make predictions on the training set (x_train)

    The function outputs the following:
    -   Y_hat (1-D list): list containing the KNN regressed values for the 
    x_test data set or x_train data setaccording to the model training.
    '''

    N_size = np.minimum(len(x_train), N_size)
    if predict_test: #If we want to make predictions on the x_test
        x_i = [[x] for x in x_train]
        KNN = KNeighborsRegressor(N_size).fit(x_i, Y_train)
        x_i = [[x] for x in x_test]
        Y_hat = KNN.predict(x_i)
    elif not predict_test: #If we want to make predictions on the x_train, this is used in the TE calculation
        x_i = [[x] for x in x_train]
        KNN = KNeighborsRegressor(N_size).fit(x_i, Y_train)
        Y_hat = KNN.predict(x_i)
    return Y_hat


# %%
N_test = 300
N_train = 50
N_sizes_plot = [1,2,5,35] #Plot these neighborhood sizes
f_x = np.sin(2.*np.pi*X_test)

for n in range(1,36):
    ge = 0
    me = 0
    y_hat = KNN_edited(N_size=n, x_train=X_train, Y_train=y_train, x_test=X_test, predict_test=True)
    for j in range(0,N_test):
        ge+=((y_test[j]-y_hat[j])**2) #Generalization error calculation
        me+=((f_x[j]-y_hat[j])**2) #Modeling error calculation

    me = me/N_test
    ge = ge/N_test
    te = 0
    
    y_hat_TE = KNN_edited(N_size=n, x_train=X_train, Y_train=y_train, x_test=X_test, predict_test=False)
    for k in range(0,N_train):
        te+=((y_train[k] - y_hat_TE[k])**2) #Training error calculation
        #print(np.shape(y_hat))
    te = te/N_train
    
    plt.plot(n, ge, '.b')
    plt.plot(n, me, '.r')
    plt.plot(n, te, '.g')
    plt.title('Generalization Error, Modeling Error, and Training Error')
    plt.ylabel('Error')
    plt.xlabel('Neighborhood Size')
    plt.legend(['GE', 'ME', 'TE'])
    #plt.show()


