import json
import matplotlib.pyplot as plt

# read the data from the file
with open('result.json', 'r') as f:
    data = json.load(f)

# extract the scores
loss = data['loss']
exact_match = [x/100 for x in data['exact match']]
f1_score = data['f1 score']

# create the plot
epochs = range(1, len(loss) + 1)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.plot(epochs, loss, label='loss', marker='o', color=(1/255, 155/255, 152/255))

# plt.plot(epochs, f1_score, label='f1 score')

# add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Loss(%)')
plt.title('Learning curve of the loss value')
plt.legend()
plt.show()
plt.savefig('loss.png')
# show the plot

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.plot(epochs, exact_match, label='exact match', marker='o', color=(1/255, 155/255, 152/255))
# add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Exact Match(%)')
plt.title('Learning curve of the Exact Match metric value')
plt.legend()
plt.show()
plt.savefig('exact_match.png')