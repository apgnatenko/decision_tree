import numpy as np


class DecisionTree():
    def __init__(self):
        self.tree = []

    def _compute_entropy(self, y):
        """
        Computes the entropy for 
        
        Args:
        y (ndarray): Numpy array indicating whether each example at a node is
            edible (`1`) or poisonous (`0`)
        
        Returns:
            entropy (float): Entropy at that node
            
        """
        entropy = 0
        if len(y)!=0:
            p1 = len(y[y==1])/len(y)
            if p1!=0 and p1!=1:   
                entropy = -p1*np.log2(p1)-(1-p1)*np.log2(1-p1)  
            else:
                entropy = 0    
        return entropy

    def _split_dataset(self, X, node_indices, feature):
        """
        Splits the data at the given node into
        left and right branches
        
        Args:
            X (ndarray):             Data matrix of shape(n_samples, n_features)
            node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
            feature (int):           Index of feature to split on
        
        Returns:
            left_indices (list):     Indices with feature value == 1
            right_indices (list):    Indices with feature value == 0
        """
        left_indices = []
        right_indices = []
        
        for node in node_indices:
            if X[node][feature]==1:
                left_indices.append(node)
            else:
                right_indices.append(node)
            
        return left_indices, right_indices

    def _compute_information_gain(self, X, y, node_indices, feature):
        
        """
        Compute the information of splitting the node on a given feature
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
    
        Returns:
            cost (float):        Cost computed
        
        """    
        # Split dataset
        left_indices, right_indices = self._split_dataset(X, node_indices, feature)
        
        X_node, y_node = X[node_indices], y[node_indices]
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        
        # Compute imformation gain
        information_gain = self._compute_entropy(y_node) - ((len(X_left)/len(X_node))*self._compute_entropy(y_left)+\
                                                    (len(X_right)/len(X_node))*self._compute_entropy(y_right))    
        return information_gain

    def _get_best_split(self, X, y, node_indices):   
        """
        Returns the optimal feature and threshold value
        to split the node data 
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

        Returns:
            best_feature (int):     The index of the best feature to split
        """    
        num_features = X.shape[1]

        if len(set(y.flatten()))==1:
            return -1
        
        best_feature = -1
        max_information_gain = 0
        for feature in range(num_features):
            current_information_gain = self._compute_information_gain(X, y, node_indices, feature)
            if current_information_gain > max_information_gain:
                max_information_gain = current_information_gain
                best_feature = feature

        return best_feature

    def build_tree_recursive(self, X, y, node_indices, branch_name, max_depth, current_depth):
        """
        Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
        This function just prints the tree.
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
            max_depth (int):        Max depth of the resulting tree. 
            current_depth (int):    Current depth. Parameter used during recursive call.
    
        """ 
        # Maximum depth reached - stop splitting
        if current_depth == max_depth:
            formatting = " "*current_depth + "-"*current_depth
            print(formatting, "%s leaf node with indices" % branch_name, node_indices)
            return
    
        # Otherwise, get best split and split the data
        # Get the best feature and threshold at this node
        best_feature = self._get_best_split(X, y, node_indices) 
        
        formatting = "-"*current_depth
        print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
        
        # Split the dataset at the best feature
        left_indices, right_indices = self._split_dataset(X, node_indices, best_feature)
        self.tree.append((left_indices, right_indices, best_feature))
        
        # continue splitting the left and the right child. Increment current depth
        self.build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
        self.build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)