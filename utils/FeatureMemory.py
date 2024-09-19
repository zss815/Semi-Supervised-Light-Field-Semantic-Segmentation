import torch
import numpy as np


class PixelFeatureMemory:
    def __init__(self, num_labeled_samples, num_classes=14, num_per_class=256):
        self.num_labeled_samples = num_labeled_samples
        self.num_per_class = num_per_class
        self.memory = [None] * num_classes
        self.num_per_class_per_image = max(1, int(round(num_per_class / num_labeled_samples)))
        self.num_classes = num_classes

    def add_features(self, features, labels, confs, batch_size):
        """
        Args:
            features: NxF feature vectors
            labels: N corresponding labels to the [features]
            confs: N corresponding confidences to the [features]
        """
        features = features.detach()  #[N,C]
        labels = labels.detach()  #[N]
        confs=confs.detach()  #[N]

        elements_per_class = batch_size * self.num_per_class_per_image

        #For each class, save [elements_per_class]
        for c in range(self.num_classes):
            mask_c = (labels == c)  # Get mask for class c
            features_c = features[mask_c, :] # [n,F]
            confs_c=confs[mask_c]  #[n]
            
            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    # Sort according to the confidences
                    _, indices = torch.sort(confs_c)
                    indices = indices.cpu().numpy()
                    features_c = features_c.cpu().numpy()
                    # Get features with highest rankings
                    features_c = features_c[indices, :]
                    new_features = features_c[:elements_per_class, :]
                else:
                    new_features = features_c.cpu().numpy()

                if self.memory[c] is None: # Empty
                    self.memory[c] = new_features

                else: # Add elements to the already existing list
                    # Keep only most recent memory_per_class features
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.num_per_class, :]


class ObjectFeatureMemory:
    def __init__(self, num_classes=14, num_per_class=256):
        self.num_per_class = num_per_class
        self.memory = [None] * num_classes
        self.memory_conf=[None] * num_classes
        self.memory_select=[None] * num_classes
        self.num_classes = num_classes

    def add_features(self, features, labels, confs):
        """
        Args:
            features: NxF feature vectors
            labels: N corresponding labels to the [features]
            confs: N corresponding confidences to the [features]
        """
        features = features.detach()  #[N,F]
        labels = labels.detach()  #[N]
        confs=confs.detach()  #[N]

        for c in range(self.num_classes):
            mask_c = (labels == c)  # Get mask for class c
            features_c = features[mask_c, :] # [n,F]
            confs_c=confs[mask_c]  #[n]
            
            if features_c.shape[0] > 0:
                features_c = features_c.cpu().numpy()
                confs_c=confs_c.cpu().numpy()

                if self.memory[c] is None: # Empty
                    self.memory[c] = features_c
                    self.memory_conf[c]=confs_c
    
                else: # Add elements to the already existing list
                    # Keep only most recent memory_per_class features
                    self.memory[c] = np.concatenate((features_c, self.memory[c]), axis = 0)
                    self.memory_conf[c] = np.concatenate((confs_c, self.memory_conf[c]), axis = 0)
    
    def output_features(self):
        for c in range(self.num_classes):
            features_c=self.memory[c]
            if features_c is not None:
                num=features_c.shape[0]
                idx=np.random.permutation(num)
                features_c=features_c[idx]
                self.memory_select[c]=features_c[:self.num_per_class]
            
        return self.memory_select