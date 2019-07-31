import numpy as np 
    
class Conv2Linear:
    """
    An intermidate layer to transform convolutional layers data into Linear layers data.
    """
    
    def forward(self, previous_layer_output):
        """
        Reshape the (N, C, H, W) input into (C*H*W, N)
        
        Inputs:
        - previous_layer_output: Input data of shape (N, C, H, W).
        Returns:
        - reshaped_output: Output data of shape (C*H*W, N).
        """
        self.N, self.C, self.H, self.W = previous_layer_output.shape
        
        reshaped_output = previous_layer_output.reshape(self.N, -1).transpose()
        
        return reshaped_output
    
    def backward(self, da):
        """
        Reshape input gradient of shape (C*H*W, N)  into (N, C, H, W) shape gradient.
        
        Inputs:
        - da - Input gradient of shape (C*H*W, N).
        
        Output:
        - Gradient of shape (N, C, H, W).
        
        """
        return da.transpose().reshape(self.N, self.C, self.H, self.W)
        
    def weight_update(self):
        pass
    

        