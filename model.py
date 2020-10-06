import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        
        super().__init__()
        
        # set up variables
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # define model layers
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        #Initialise embedding layer with a xavier uniform distribution.
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.embed.weight)
    
    def forward(self, features, captions):
        
        captions = captions[:, :-1] 
        
        # Pass through word embedding layer.
        captions = self.embed(captions)
        
        # Concatenate the image features with the embedded caption.
        # Features has shape (batch_size, embed_size)
        # but captions has shape (batch_size, caption_length, embed_size)
        # So we use unsqueeze on features to add a new dimension in position 1, so that they can stack properly.
        
        pairs = torch.cat((features.unsqueeze(1), captions), 1)
        
        # Pass through LSTM and keep output. We don't need the hidden layer.
        output, _ = self.lstm(pairs)
        
        # Finally pass through linear layer.
        output = self.fc(output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # Where caption will be stored.
        caption = []
        
        for i in range(max_len):
            
            # Pass inputs and states (starts as None because there's nothing to remember) into LSTM
            output, states = self.lstm(inputs, states)
            
            
            # Pass through linear layer. Squeeze to make sizes appropriate.
            output = self.fc(output).squeeze(1)
            
            # Choose the word ID with the highest probability andadd it to caption
            predicted_word = output.argmax(dim=1)
            caption.append(predicted_word.item())
            
            # If end token is reached earlier than max length, stop iterating.
            if predicted_word.item() == 1:
                break
            
            # Set input for next iteration as the output of this iteration. Unsqueeze to make dimensions match.
            inputs = self.embed(predicted_word.unsqueeze(0))
            
        return caption