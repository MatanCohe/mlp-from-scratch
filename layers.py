import numpy as np 
from convolution import convolution, backword

class Conv2Linear:
    """
    An intermidate layer to transform convolutional layers data into Linear layers data.
    """
    
    def forward(self, previous_layer_output, is_training=False):
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
        - Gradient of shape (N, C, H, W)
        
        """
        return da.transpose().reshape(self.N, self.C, self.H, self.W)
        
    def weights_update(self, learning_rate, l2_lambda, batch_size):
        pass
    


class MaxPool2d:
    """
    Maxpool layer that works on square blocks and no overlap.
    """
    
    def __init__(self, size):
        """
        Construct the MaxPool layer with with the given size.
        """
        self.h, self.w = size, size
    
    def forward(self, previous_layer_output, is_training=False):
        """
        Apply maxpool over the previous_layer_output.
        
        Input:
        - previous_layer_output: input of shape (N, C, H, W)
        
        Output:
        - out: pooled output of shape (N, C, H/size, W/size)
        """
        N, C, H, W = previous_layer_output.shape
        pool_height, pool_width = self.h, self.w
        
        assert pool_height == pool_width, 'Invalid pool params'
        assert H % pool_height == 0
        assert W % pool_height == 0
        x_reshaped = previous_layer_output.reshape(N, C, int(H / pool_height), pool_height, 
                                                   int(W / pool_width), pool_width)
        out = x_reshaped.max(axis=3).max(axis=4)

        self.prev_a, self.a_reshaped, self.out = previous_layer_output, x_reshaped, out
        return out
    
    def backward(self, da):
        """
        Apply unpool over the gradient.
        
        Input:
        - da: gradient of shape (N, C, H/size, W/size)
        
        output:
        - dx: gradient of shape (N, C, H, W)
        
        """
        x, x_reshaped, out = self.prev_a, self.a_reshaped, self.out

        dx_reshaped = np.zeros_like(x_reshaped)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (x_reshaped == out_newaxis)
        da_newaxis = da[:, :, :, np.newaxis, :, np.newaxis]
        da_broadcast, _ = np.broadcast_arrays(da_newaxis, dx_reshaped)
        dx_reshaped[mask] = da_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(x.shape)

        return dx
    
    def weights_update(self, learning_rate, l2_lambda, batch_size):
        pass


class Conv2d:
    
    def __init__(self, kernels, bias, activation_function, activation_function_derivative):
        self.kernels, self.b = kernels, bias 
        self.activation = activation_function
        self.activation_derivative = activation_function_derivative
    
    def forward(self, previous_layer_output, is_training=False):
        self.prev_a = previous_layer_output
        return convolution.conv4d(previous_layer_output, self.kernels)
    
    def backward(self, da):
        new_da, self.dw, self.db = backword.backward_conv5d(self.prev_a, da, self.kernels)
        return new_da
    
    def weights_update(self, learning_rate, l2_lambda, batch_size):
        self.kernels = self.kernels - learning_rate * self.dw
        self.b = self.b - learning_rate * self.db