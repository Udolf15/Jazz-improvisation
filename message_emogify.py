from keras.models import load_model
import numpy as np
import argparse 
import codecs
from emo_utils import *

# load save model
model = load_model('model/emojifier_model.h5')

# load glove embedding model
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--text", type=str,
	help="Text file name")
args = vars(ap.parse_args())

# Open the file
f = open("input/"+args['text'], "r")

# Convert the opened file to string format
strI = f.read()

f.close()

# Convert the sentences to array 

oldSentenceArray = strI.split(".")
newSentenceArray = []

# maxLen of sentence depend upon training data
maxLen = 10

for sentence in oldSentenceArray:
    newSentenceArray.append(sentence.strip().lower())


# Creating a numpy array
npSentenceArray = np.array(newSentenceArray)

X_sentences_indices = sentences_to_indices(npSentenceArray, word_to_index, maxLen)

emoArr = np.argmax(model.predict(X_sentences_indices) , axis = 1)
finalOutput = ""

# Creating the final string to be stored as output

for i in range(emoArr.shape[0]):
    if(len(oldSentenceArray[i]) == 0):
        continue
    finalOutput = finalOutput + oldSentenceArray[i]+ " " + label_to_emoji(emoArr[i]) + " ."

# Write the final string to emogified_message.txt

fileI = codecs.open("output/emogified_message.txt", "w", "utf-8")
fileI.write(finalOutput)
fileI.close()

print("Message Emogified")