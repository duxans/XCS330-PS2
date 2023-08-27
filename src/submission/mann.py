import torch
from torch import nn, Tensor
import torch.nn.functional as F


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        ### START CODE HERE ###
   
        K = self.samples_per_class - 1
        B = input_images.shape[0]
        N = self.num_classes

        #Concatenate the set of labels and images and pass zeros for the final K+1th prediction      
        concatenated_input = torch.cat((input_images.view(B, -1, 784), input_labels.view(B, -1, N)), dim=2)
        concatenated_input = concatenated_input.view(B, K+1, N, 784+N)
        
        #Pass zeros for the final query prediction
        concatenated_input[:, K,:,784:]= torch.zeros(B, N, N)

        #Reshape for LSTM layer1 (batch, seq_len, feature)
        concatenated_input = concatenated_input.view(B,(K+1)*N,784+N)
        
        output,_ = self.layer1(concatenated_input)
        output,_ = self.layer2(output)

        #Reshape the output to get the predictions as Description
        predictions = output.view(B, K+1, N, N) 
                      
        return predictions      
        ### END CODE HERE ###

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################

        loss = None

        ### START CODE HERE ### 
        K = self.samples_per_class - 1
        N = self.num_classes
        preds = preds[:,K,:]
        labels = labels[:,K,:]
      
        preds = preds.reshape(-1, N)
        labels = labels.reshape(-1, N)
        loss = F.cross_entropy(preds, labels)
        print('loss: ', loss)
        ### END CODE HERE ###

        return loss
