import argparse # Library for parsing command line arguments in calls of main method
import torch # Machine learning library
import torch.nn as nn # Sub library of neural network components
import tqdm # Verbose history printing tool
import matplotlib.pyplot as plt # Plotting library
import numpy as np # Library for managing array objects and files

# A construction of a basic classification model featuring some common assembly methods
class Classifier(nn.Module):
    def __init__(self, arg1: int, arg2: float = 0.):
        super(Classifier, self).__innit__()
        self.layers = arg1 # Argument for number of layers in the model
        self.dropout = arg2 # Argument for rate of randomly dropping one layer's value
        self.input_features = 3 # Number of variables in your input data (will depend on input data)
        
        # Define some input layer of
        linear1 = nn.Linear(in_features=self.input_features, out_features = 2**self.layers)
        
        # Initialize the embedding layers
        self.hidden_layer_blocks = nn.ModuleList()
        
        # For the layers argument, create layers of decreasing neurons until reaching an output of size 1
        # Note, this classifier assumes you only have one single class label
        for layer in self.layers:
            self.hidden_layer_blocks.append(
                nn.Dropout(p=self.dropout),
                nn.ReLU(),
                nn.Linear(in_features=4**(self.layers - layer - 1), out_features=2**(self.layers - layer)),   
            )
        
        # A layer that limits the total range of outputs for a particular type of normalization
        sigmoid = nn.Sigmoid()
    
    # The forward method constructs the model
    def forward(self, inputs):
        inputs = linear1(inputs) # Apply linear1 layer to input data
        hidden = self.hidden_layer_blocks(inputs) # Apply hidden layer blocks to input data
        outputs = sigmoid(hidden) # Apply sigma to output of hidden layer blocks
        
        return outputs
        
def train_batch(model, optimizer, train_data, labels, loss):
    # Enable default settings for pytorch training
    torch.set_default_dtype(torch.float32)
    torch.enable_grad()
    
    # Reset the optimizer
    optimizer.zero_grad()
    
    # Make a prediction from a batch
    prediction = torch.flatten(model(train_data))
    
    # Get error from the model's prediction
    error = loss(prediction, labels)
    
    # Propagate the error
    error.backward()
    
    # Update weights and biases using the optimizer
    optimizer.step()
    
    return error, prediction

def validate(model, valid_data, loss):
    # Set the model into evaluation mode
    model.eval()
    
    errors = []
    # Find the loss for the entire validation set
    for batch, labels in valid_data_loader:
        with torch.no_grad():
            prediction = torch.flatten(model(batch))
            error = loss(prediction, labels)
            errors.append(float(error))
    
    return sum(errors)


def train_model(model, loss, args, train_plot_loader, valid_plot_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    
    # Training loop
    for epoch in tqdm(range(1, epochs+1)):
        print('Epoch: ', epoch)
        errors = []
        valid_errors = []
        train_errors = []
        
        # Train and get errors from each batch
        for iteration, data in enumerate(tqdm(train_loader)):
            batch, labels = data
            error, prediction = train_batch(model, optimizer, train_data, labels, loss, epoch)
            errors.append(float(error))
        valid_errors.append(validate(model, valid_data, loss))
        train_errors.append(sum(errors))
        
        # Save model layer weights to a state dictionary
        torch.save(model.state_dict(), "{0}/Settings_Epoch_{1}".format(args.output, epoch))
    
    return train_errors, valid_errors

def plot_errors(errors, valid_errors, args):
    epochs = list(range(args.epochs))
    
    # Create plot of loss history
    ax, fig = plt.subplots()
    ax.plot(epochs, train_errors, label='Training Loss')
    ax.plot(epochs, valid_errors, label='Validation Loss')
    ax.legend()
    ax.set_title('Loss History')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    fig.savefig("{0}/Loss_History".format(args.output))
        
def main():
    # Define and read in command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg1', help='Argument of type int', type=int)
    parser.add_argument('--arg2', help='Argument of type float', type=float)
    parser.add_argument('-e', '--epochs', help='Epochs argument of type int', type=int)
    parser.add_argument('-l', '--lr', help='Learning rate argument of type float', type=float) 
    parser.add_argument('-o', '--output', help='Path argument of type string', type=str)
    parser.add_argument('-i', '--input', help='Path argument of type string', type=str)
    args = parser.parse_args()
    
    # If gpu is available, use gpu, otherwise, cpu
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    
    # Identify type of processing device
    device = torch.device(dev)
    print('Device to be used for training:')
    print(device)
    
    # Import data sets from, for example, saved numpy arrays
    # This example would be if you had separately stored signal and background data
    signal_train = np.load("{0}/signal_train.npy".format(args.input))
    signal_valid = np.load("{0}/signal_train.npy".format(args.input))
    bg_train = np.load("{0}/signal_train.npy".format(args.input))
    bg_valid = np.load("{0}/signal_train.npy".format(args.input))
    
    # Compile final datasets
    train_data = np.concatenate((signal_train, bg_train))
    valid_data = np.concatenate((signal_valid, bg_valid))
    
    # Convert to tensors for torch
    train_data = torch.tensor(train_data, dtype=torch.double)
    valid_data = torch.tensor(valid_data, dtype=torch.double)
    
    # Create the labels for the signal events
    train_labels = torch.ones(np.shape(signal_train)[0])
    valid_labels = torch.ones(np.shape(signal_valid)[0])
    
    # Add on the labels for the background events
    train_labels = torch.cat((train_labels, torch.zeros(np.shape(bg_train)[0])))
    valid_labels = torch.cat((valid_labels, torch.zeros(np.shape(bg_valid)[0])))
    
    # Create Tensor Datasets for ease of use with data loaders
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_labels)

    # Create data loaders for the plotting functions
    train_plot_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    valid_plot_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=True, drop_last=True)
    
    # Build model
    model = Classifier(arg1=arg1, arg2=arg2)
    
    # Define loss function
    # Here we use binary cross entropy loss to match our classification of 0 (background) and 1 (signal) data labels
    loss = nn.BCELoss()
    
    # Train the model
    train_errors, valid_errors = train_model(model, loss, args, train_plot_loader, valid_plot_loader)
    
    # Plot the loss history
    plot_errors(errors, valid_errors, args)

if __name__ == '__main__':
    main()