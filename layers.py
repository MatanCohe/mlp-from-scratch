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
        if is_training:
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
        z = convolution.conv4d(previous_layer_output, self.kernels, self.b)
        a = self.activation(z)
        if is_training:
            self.prev_a, self.z, self.a = previous_layer_output, z, a
        return a
    
    def backward(self, da):
        da = da * self.activation_derivative(self.z)
        new_da, self.dw, self.db = backword.backward_conv5d(self.prev_a, da, self.kernels)
        return new_da
    
    def weights_update(self, learning_rate, l2_lambda, batch_size, beta=0.9):
        dw, db = self.dw, self.db
        prev_vw = self.vw if hasattr(self, 'vw') else np.zeros_like(dw)
        prev_vd = self.vb if hasattr(self, 'vb') else np.zeros_like(db)
        vw = beta * prev_vw + (1-beta) * dw 
        vb = beta * prev_vd + (1-beta) * db
        self.kernels = self.kernels - learning_rate * vw - l2_lambda * learning_rate * self.kernels
        self.b = self.b - learning_rate * vb
        self.vw, self.vb = vw, vb
        self.dw, self.db = None, None
        
from cs231n import fast_layers
class BatchNorm2d:
    
    def __init__(self, gamma, beta, activation_function, activation_function_derivative):
        self.gamma, self.beta = gamma, beta
        self.activation = activation_function
        self.activation_derivative = activation_function_derivative
    
    def forward(self, previous_layer_output, is_training=False):
        self.prev_a = previous_layer_output
        N, C, H, W = previous_layer_output.shape
        reshaped = previous_layer_output.reshape(N, -1)
        X, norm, mu, var = self._batchnorm_forward(reshaped)
        z = X.reshape(N, C, H, W)
        a = self.activation(z)
        if is_training:
            self.z, self.a, self.X, self.X_norm, self.mu, self.var = z, a, reshaped, norm, mu, var
        return a
        
    
    def backward(self, da):
        N, C, H, W = da.shape
        da = da * self.activation_derivative(self.z)
        da_reshaped = da.reshape(N, -1)
        dX = self._batchnorm_backword(da_reshaped)
        dx_reshaped = dX.reshape(N, C, H, W)
        return dx_reshaped
    
    def weights_update(self, learning_rate, l2_lambda, batch_size, beta=0.9):
        dgamma, dbeta = self.dgamma, self.dbeta
        
        self.gamma, self.beta = self.rmsprop(dgamma, dbeta, learning_rate, l2_lambda, batch_size, beta)
        self.dgamma, self.dbeta = None, None
    
    def sgd_with_momentum_update(self, dw, db, learning_rate, l2_lambda, batch_size, beta=0.9):
        theta, b, alpha = self.gamma, self.beta, learning_rate
        prev_vw = self.vw if hasattr(self, 'vw') else np.zeros_like(dw)
        prev_vd = self.vb if hasattr(self, 'vb') else np.zeros_like(db)
        vw = beta * prev_vw + (1-beta) * dw
        vb = beta * prev_vd + (1-beta) * db
        new_w = theta - alpha * vw - l2_lambda * alpha * theta
        new_b = b - alpha * vb
        self.vw, self.vb = vw, vb
        return new_w, new_b
        
    def rmsprop(self, dw, db, learning_rate, l2_lambda, batch_size, beta=0.9):
        theta, b, alpha = self.gamma, self.beta, learning_rate
        prev_vw = self.vw if hasattr(self, 'vw') else np.zeros_like(dw)
        prev_vd = self.vb if hasattr(self, 'vb') else np.zeros_like(db)
        eps = 1e-8
        vw = beta * prev_vw + (1-beta) * dw * dw
        vb = beta * prev_vd + (1-beta) * db * db
        new_w = theta - np.divide(alpha, np.sqrt(vw+eps)) * dw
        new_b = b - np.divide(alpha, np.sqrt(vb+eps)) * db
        self.vw, self.vb = vw, vb
        return new_w, new_b
        
    
    
    def _batchnorm_forward(self, X):
        gamma, beta = self.gamma, self.beta
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_norm = (X - mu) / np.sqrt(var + 1e-8)
        out = gamma * X_norm + beta

        return out, X_norm, mu, var
    
    def _batchnorm_backword(self, da):
        X, X_norm, mu, var, gamma, beta = self.X, self.X_norm, self.mu, self.var, self.gamma, self.beta

        N, D = X.shape

        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + 1e-8)

        dX_norm = da * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(da * X_norm, axis=0)
        dbeta = np.sum(da, axis=0)
        
        self.dgamma, self.dbeta = dgamma, dbeta

        return dX


class FastConv:
    
    def __init__(self, kernels, bias, activation_function, activation_function_derivative):
        self.kernels, self.b = kernels, bias 
        self.activation = activation_function
        self.activation_derivative = activation_function_derivative
        self.params = {
            'pad': 1, 
            'stride': 1,
        }
        
        
    def forward(self, previous_layer_output, is_training=False):
        z, cache = fast_layers.conv_forward_im2col(previous_layer_output, self.kernels, self.b, self.params)
        if is_training:
            self.z, self.cache = z, cache
        return self.activation(z)
    
    def backward(self, da):
        delta = da * self.activation_derivative(self.z)
        dx, self.dw, self.db = fast_layers.conv_backward_im2col(delta, self.cache)
        return dx
        
    
    def weights_update(self, learning_rate, l2_lambda, batch_size, beta=0.9):
        dw, db = self.dw, self.db
        self.kernels, self.b = self.rmsprop(dw, db, learning_rate, l2_lambda, batch_size, beta=0.9)
        self.dw, self.db = None, None

    def sgd_with_momentum_update(self, dw, db, learning_rate, l2_lambda, batch_size, beta=0.9):
        theta, b, alpha = self.kernels, self.b, learning_rate
        prev_vw = self.vw if hasattr(self, 'vw') else np.zeros_like(dw)
        prev_vd = self.vb if hasattr(self, 'vb') else np.zeros_like(db)
        vw = beta * prev_vw + (1-beta) * dw
        vb = beta * prev_vd + (1-beta) * db
        new_w = theta - alpha * vw - l2_lambda * alpha * theta
        new_b = b - alpha * vb
        self.vw, self.vb = vw, vb
        return new_w, new_b
        
    def rmsprop(self, dw, db, learning_rate, l2_lambda, batch_size, beta=0.9):
        theta, b, alpha = self.kernels, self.b, learning_rate
        prev_vw = self.vw if hasattr(self, 'vw') else np.zeros_like(dw)
        prev_vd = self.vb if hasattr(self, 'vb') else np.zeros_like(db)
        eps = 1e-8
        vw = beta * prev_vw + (1-beta) * dw * dw
        vb = beta * prev_vd + (1-beta) * db * db
        new_w = theta - np.divide(alpha, np.sqrt(vw+eps)) * dw
        new_b = b - np.divide(alpha, np.sqrt(vb+eps)) * db
        self.vw, self.vb = vw, vb
        return new_w, new_b
        
class Dropout:
    
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        
    def forward(self, previous_layer_output, is_training=False):
        if is_training:
            mask = np.random.rand(*previous_layer_output) < (1 - self.dropout_rate)
        else:
            mask = (1 - self.dropout_rate)
        return mask * previous_layer_output
    
    def backward(self, da):
        return self.mask * da
        
    
    def weights_update(self, learning_rate, l2_lambda, batch_size, beta=0.9):
        self.mask = None
        