"""
Since the outputs from the generator are not decoded and re-encoded when being fed to the discriminator,
all models need to use the same pretrained embedding layer to be able to understand each other.
This code is pretty much just a modification from 
https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#word-embeddings-in-pytorch
"""
import torch
import torch.nn as nn
from Dataset import Dataset
from tqdm import trange

def closest(target, n=10):
	cos = nn.CosineSimilarity(dim=0)
	top_n_words = []
	top_n_values = [0 for _ in range(n)]
	for word in dataset.unique_tokens:
		embedded = embedding(torch.tensor(dataset.word_to_ix[word]))
		similarity = cos(target, embedded).item()
		if similarity > top_n_values[-1] or len(top_n_words) < n:
			top_n_values[-1] = similarity
			top_n_values.sort(reverse=True)
			top_n_words.insert(top_n_values.index(similarity), word)

	for i in range(n):
		print(f"{top_n_values[i]}: {top_n_words[i]}")

	return top_n_words, top_n_values

# define dataset
dataset = Dataset(path="data/small_dataset.txt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

embedding = nn.Embedding(len(dataset.unique_tokens), 50)

class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = Model(len(dataset.unique_tokens), 50, 3)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in trange(10):
    total_loss = 0
    for context, target in samples:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([dataset.word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!

closest(embedding(torch.tensor(dataset.word_to_ix["war"])))

