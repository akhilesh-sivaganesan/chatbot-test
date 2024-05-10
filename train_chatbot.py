import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize lists to store words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Preprocess intents and patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Add documents in the corpus
        documents.append((word, intent['tag']))
        # Add intent tag to classes list if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words, convert to lowercase, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes to pickle files
with open('words.pkl', 'wb') as file:
    pickle.dump(words, file)
with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

# Create training data
training = []
output_empty = [0] * len(classes)

# Generate bag of words and one-hot encode output for each document
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert training data to numpy array
random.shuffle(training)
training = np.array(training)

# Split into input (X) and output (Y) arrays
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Define optimizer with learning rate
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile model with specified optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save model
model.save('chatbot_model.h5')

print("Model created and saved successfully.")
