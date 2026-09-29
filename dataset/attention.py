import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Open the text file
file = open("dataset/sample_sentences.txt", "r")

# Read all sentences
sentences = file.readlines()

# Close the file
file.close()
# Display sentences
for sentence in sentences:
    print(sentence.strip())

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the sentences
embeddings = model.encode(sentences)

# Save the embeddings to a .npy file
np.save("dataset/embeddings.npy", embeddings)

print("Embeddings saved successfully.")

# create a dataframe with the sentences and their corresponding embeddings
df = pd.DataFrame({'sentence': sentences, 'embedding': list(embeddings)})

# load the embeddings from the .npy file
loaded_embeddings = np.load("dataset/embeddings.npy")

print("Embedding shape:", loaded_embeddings.shape)
# create Q, K, V matrices for attention mechanism
embedding_size = loaded_embeddings.shape[1]

W_Q = np.random.rand(embedding_size, embedding_size)
W_K = np.random.rand(embedding_size, embedding_size)

W_V = np.random.randn(embedding_size, embedding_size)
Q = loaded_embeddings @ W_Q
K = loaded_embeddings @ W_K
V = loaded_embeddings @ W_V
scores = Q @ K.T
print("Attention Scores:")
print(scores)
#Scaled Dot-Product Attention
d_k = K.shape[1]
scaled_scores = scores / np.sqrt(d_k)
print(scaled_scores)
#Implement Softmax Manually
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
attention_weights = softmax(scaled_scores)
print("Attention Weights:")
print(attention_weights)
print(np.sum(attention_weights, axis=1))
#Calculate Final Attention Output
final_output = attention_weights @ V
print("Final Attention Output:")
print(final_output)

#display all the matrices
print("=" * 50)
print("QUERY (Q)")
print("=" * 50)
print(Q)
print("\nKEY (K)")
print("=" * 50)
print(K)
print("\nVALUE (V)")
print("=" * 50)
print(V)
print("\nATTENTION SCORES")
print("=" * 50)
print(scaled_scores)
print("\nATTENTION WEIGHTS")
print("=" * 50)
print(attention_weights)
print ("\nATTENTION OUTPUT")
print("=" * 50)
print(final_output)

#save the attention weights to a CSV file
weights_df = pd.DataFrame(attention_weights)
weights_df.to_csv("dataset/attention_weights.csv",index=False)
print("Attention weights saved successfully.")
#save the final attention output to a CSV file 
output_df = pd.DataFrame(final_output) 
output_df.to_csv("dataset/attention_output.csv", index=False)
print ("Attention output saved successfully.")
#Visualize Attention Using Heatmap

plt.figure(figsize=(8, 6))

sns.heatmap(
    attention_weights,
    annot=True,
    fmt=".2f",
    cmap="Purples"
)
plt.title("Attention Weights Heatmap")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.tight_layout()
plt.savefig("dataset/attention_heatmap.png")
plt.show()