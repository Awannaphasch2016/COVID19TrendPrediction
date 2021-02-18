""" testing humiingbird """

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hummingbird.ml import convert
from Utils.utils import timer

# Create some random data for binary classification
num_classes = 2
X = np.random.rand(100000, 28)
y = np.random.randint(num_classes, size=100000)

# Create and train a model (scikit-learn RandomForestClassifier in this case)
skl_model = RandomForestClassifier(n_estimators=10, max_depth=10)
skl_model.fit(X, y)

# Use Hummingbird to convert the model to PyTorch
model = convert(skl_model, 'pytorch')

# Run predictions on CPU
model.predict(X)
print(timer(model.predict, X))

# Run predictions on GPU
model.to('cuda')
model.predict(X)
print(timer(model.predict, X))

# # Save the model
# model.save('hb_model')

# # Load the model back
# model = hummingbird.ml.load('hb_model')
# print(timer(model.predict, X))
