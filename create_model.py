from model import *

# Defining data

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')
maxLen = len(max(X_train, key=len).split())

# Using glove embedding model
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


# Initializing the model and fitting the model

model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)
model.summary()
model.fit(X_train_indices, Y_train_oh, epochs = 60, batch_size = 32, shuffle=True)

# Saving the model

model.save('model/emogifier_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
