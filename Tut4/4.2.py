import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

# Load csv file into the dataframe: df
df = pd.read_csv("titanic_all_numeric.csv")

# Convert the boolean values of the 'age_was_missing' column to integer
df.age_was_missing = df.age_was_missing.replace({True: 1, False: 0})

# Create predictors NumPy array: predictors
predictors = df.drop(['survived'], axis=1).values

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Convert the target to categorical: target
target = to_categorical(df['survived'])

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def get_new_model():
  # Set up the model
  model = Sequential()
  model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
  model.add(Dense(2, activation='softmax'))

  return model
# Import the SGD optimizer
from tensorflow.keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.01, 1.0]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n' % lr)

    # Build new model to test, unaffected by previous models
    model = get_new_model()

    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=my_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model
    model.fit(predictors, target, epochs=10, batch_size=32, verbose=1)
