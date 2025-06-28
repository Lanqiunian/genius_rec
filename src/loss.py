# src/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SampledSoftmaxLoss(nn.Module):
    """
    Sampled Softmax Loss for models with large output vocabularies.
    This implementation is based on the common practice in recommender systems.
    """
    def __init__(self, num_sampled, num_classes, nv_noise=None):
        """
        Args:
            num_sampled (int): The number of negative samples to use.
            num_classes (int): The total number of classes (items).
            nv_noise: Noise distribution for negative sampling. Not used in this version.
        """
        super(SampledSoftmaxLoss, self).__init__()
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        
        # We will use the item embedding weights for the softmax layer
        # This parameter will be tied to the model's item embedding layer
        # Here we initialize it, but it should be overwritten in the training loop
        # with the actual embedding weights.
        self.softmax_w = nn.Parameter(torch.Tensor(num_classes, 1))
        self.softmax_b = nn.Parameter(torch.Tensor(num_classes))

    def forward(self, inputs, labels, neg_samples):
        """
        Args:
            inputs (torch.Tensor): The output of the model. Shape: [B, L, D]
            labels (torch.Tensor): The ground truth positive item ids. Shape: [B, L]
            neg_samples (torch.Tensor): The sampled negative item ids. Shape: [B, L, num_sampled]
        Returns:
            torch.Tensor: The computed loss.
        """
        # Flatten inputs and labels to treat each position independently
        inputs = inputs.view(-1, inputs.size(-1))  # [B*L, D]
        labels = labels.view(-1)  # [B*L]
        neg_samples = neg_samples.view(-1, self.num_sampled) # [B*L, num_sampled]

        # Get the embeddings/weights for the positive and negative samples
        # from the model's main embedding layer.
        # This is a crucial step: the weights of the loss function's output layer
        # are tied to the input embedding weights of the model.
        
        # We assume the tied weights are in the `self.softmax_w` which should be
        # set to `model.item_embedding.weight` outside this class.
        
        # Get positive sample weights and biases
        true_w = self.softmax_w[labels] # [B*L, D]
        true_b = self.softmax_b[labels] # [B*L]
        
        # Get negative sample weights and biases
        sampled_w = self.softmax_w[neg_samples] # [B*L, num_sampled, D]
        sampled_b = self.softmax_b[neg_samples] # [B*L, num_sampled]

        # Calculate logits
        # Logit for the positive class
        true_logits = (inputs * true_w).sum(dim=1) + true_b
        
        # Logits for the negative classes
        # Unsqueeze inputs for broadcasting: [B*L, 1, D]
        sampled_logits = (inputs.unsqueeze(1) * sampled_w).sum(dim=2) + sampled_b
        
        # Concatenate true and sampled logits
        # Resulting shape: [B*L, 1 + num_sampled]
        logits = torch.cat([true_logits.unsqueeze(1), sampled_logits], dim=1)
        
        # The target for cross-entropy is always the first class (the positive one)
        # Target shape: [B*L]
        pseudo_labels = torch.zeros_like(labels)
        
        # We only care about positions that are not padding
        # Assuming padding token ID is 0 for labels
        loss_mask = (labels != 0).float()
        
        loss = F.cross_entropy(logits, pseudo_labels, reduction='none')
        
        # Apply mask and calculate the mean loss
        loss = (loss * loss_mask).sum() / loss_mask.sum()
        
        return loss

    def tied_weight(self, weight, bias):
        """
        A method to tie the weights of the model's item embedding to this loss function.
        This should be called in the training loop before calculating the loss.
        """
        self.softmax_w = weight
        self.softmax_b = bias