import numpy as np 
    
class Conv2Linear:
    
    def forward(self, previous_layer_output):
        
        self.N, self.C, self.H, self.W = previous_layer_output.shape
        
        reshaped_output = previous_layer_output.reshape(self.N, -1).transpose()
        
        return reshaped_output
    
    def backward(self, da):
        return da.transpose().reshape(self.N, self.C, self.H, self.W)
        
    def weight_update(self):
        pass
    

        