# Generic-Classifier
## An example of a generic pytorch classifier model

### Dependencies:
argparse
pytorch
tqdm
matplotlib
numpy (should be installed by pytorch)

### Usage instructions:
1. Install python environment with necessary dependencies
2. Activate python enviroment
3. Use command line to train model e.g.
python3 main --arg1 4 --arg2 .2 -e 20 -l .0001 -i /Users/username/data -o /Users/username/results

### Command line arguments:
--arg1 : number of model layers

--arg2 : dropout rate (value from 0.-1.

--epochs (-e) : number of epochs of training

--lr (-l) : learning rate

--output (-o) : output file path

--input (-i) : input file path
