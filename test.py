# What version of Python do you have?
import sys
import platform
import torch
import pandas as pd
import sklearn as sk

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


pokedex = pd.read_csv('data/Pokedex_Cleaned.csv', encoding='latin-1')

primary_types = pokedex['Primary Type']
primary_types.value_counts()  # need to remove rows that aren't types

not_types = {"Null", "Male", "Female", "Rockruff"}
pokedex = pokedex.loc[~pokedex['Primary Type'].isin(not_types)]  # remove rows with invalid types
primary_types = pokedex['Primary Type']

# Drop irrelevant rows, columns, reorder name
pokemon_names = pokedex['Name'].copy()
pokedex_relevant = pokedex.drop(['#', 'Name', 'Secondary Type', 'Total', 'Variant'], axis=1)
pokedex_relevant['Name'] = pokemon_names
pokedex_relevant = pokedex_relevant.drop_duplicates(subset=["Name"], keep='last')

pokedex_relevant.Name.tolist()

# extract features and labels
features = pokedex_relevant.iloc[:, 1:-1].values.astype(float)
# normalize numbers
for feature in features:
    s = sum(feature)
    feature /= s
labels = pokedex_relevant.iloc[:, 0].values

# train test splits -- stats
# Stratify on labels to ensure each class has equal proportions
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, stratify=labels, random_state=42)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        # Using the nn.Sequential function, define a 5 layers of convolutions
        self.conv_relu_stack =  nn.Sequential(nn.Conv2d(3,32, kernel_size=3, padding=1),
                                              nn.LeakyReLU(),
                                              nn.MaxPool2d(kernel_size=2),
                                              nn.Conv2d(32,64, kernel_size=3, padding=1),
                                              nn.LeakyReLU(),
                                              nn.MaxPool2d(kernel_size=2),
                                              nn.Conv2d(64,128, kernel_size=3, padding=1),
                                              nn.LeakyReLU(),
                                              nn.MaxPool2d(kernel_size=2),
                                              nn.Conv2d(128,256, kernel_size=3, padding=1),
                                              nn.LeakyReLU(),
                                              nn.MaxPool2d(kernel_size=2),
                                              nn.Conv2d(256,512, kernel_size=3, padding=1),
                                              nn.LeakyReLU())
        # with ReLU activation function
        # with sizes of input and output nodes as:
        # layer1: 1,32 , kernel size of 3, with padding of 1
        # layer2: 32, 64, kernel_size 3, with padding of 1
        # pooling layer 
        # with ReLU activation function
        # you can add more conv, pooling and relu layers
        # Last layer: in:512, out: 10 (for 10 output classes)
        self.linear = nn.Sequential(nn.Linear(2048,512),
                                    nn.LeakyReLU(),
                                    nn.Linear(512,18),
                                    nn.LeakyReLU())

    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits

model = NeuralNetwork().to(device)

# Use cross-entropy loss as the loss function
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3

# Define a pytorch optimizer using stochastic gradient descent (SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

epochs = 25

# for plotting the training loss
history = {'losses': [], 'accuracies': []}
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    history['losses'].append(train(train_dataloader, model, loss_fn, optimizer))
    history['accuracies'].append(test(test_dataloader, model, loss_fn))
    
    plt.clf()
    fig1 = plt.figure()
    plt.plot(history['losses'], 'r-', lw=2, label='loss')
    plt.legend()
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.clf()
    fig2 = plt.figure()
    plt.plot(history['accuracies'], 'b-', lw=1, label='accuracy')
    plt.legend()
#     display.clear_output(wait=True)
    display.display(plt.gcf())
print("Done!")  # likely could use more epochs

# Make predictions on random 40 test data items using the trained model and visualize them
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        selector_index = random.randint(0,len(test_data)-1)
        plt.subplot(n_rows, n_cols, index + 1)
        X, y = test_data[selector_index]
        y_pred = model(X.to(device)[None,...])
        y_pred = y_pred.argmax(1)
        plt.imshow(np.fliplr(np.rot90(X.T, k=3, axes=(0, 1))), interpolation="nearest")
        plt.axis('off')
        if y == y_pred:
            plt.title(class_names[y_pred], fontsize=12, color='b')
        else:
            plt.title(class_names[y_pred], fontsize=12, color='r')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

true_ys = []
pred_ys = []
for selector_index in range(len(test_data)):
    X, y = test_data[selector_index]
    y_pred = model(X.to(device)[None,...])
    y_pred = y_pred.argmax(1)
    y = class_names[y]
    y_pred = class_names[y_pred]
    true_ys.append(y)
    pred_ys.append(y_pred)

print("For test set:")
print("Confusion Matrix: ",
confusion_matrix(true_ys, pred_ys))

print ("Accuracy : ",
accuracy_score(true_ys, pred_ys)*100)

print("Report : ",
classification_report(true_ys, pred_ys))

true_ys = []
pred_ys = []
for selector_index in range(len(training_data)):
    X, y = training_data[selector_index]
    y_pred = model(X.to(device)[None,...])
    y_pred = y_pred.argmax(1)
    y = class_names[y]
    y_pred = class_names[y_pred]
    true_ys.append(y)
    pred_ys.append(y_pred)

print("For train set:")
print("Confusion Matrix: ",
confusion_matrix(true_ys, pred_ys))

print ("Accuracy : ",
accuracy_score(true_ys, pred_ys)*100)

print("Report : ",
classification_report(true_ys, pred_ys))