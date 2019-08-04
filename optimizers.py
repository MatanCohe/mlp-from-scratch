import numpy as np

class SGD:
    
    def __init__(batch_size,alpha=0.01, decay_rate=1, l2_lambda=0, use_avg=False):
        self.batch_size = batch_size
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.use_avg = use_avg
        self.n_calls = 0
    
    def optimize(trainable_layers):
        self.n_calls += 1
        if n_calls % self.batch_size:
            self.alpha = self.decay_rate * self.alpha
        for layer in trainable_layers:
            w = layer.w
            dw = layer.dw
            b = layer.b
            db = layer.db
            if self.use_avg:
                dw = np.divide(dw, self.batch_size)
                db = np.divide(db, self.batch_size)

            dw = dw + np.divide(self.decay_rate, self.batch_norm) * w
            
            new_dw = self._calculateNewWeight(w, dw)
            new_db = self._calculateNewWeight(b, db)
            layer.update(new_dw, new_db)
    
    def _calculateNewWeight(x, dx):
        return x - self.alpha * dx
    

class RMSProp:
    import collections
    def __init__(batch_size, alpha=0.01, beta=0.1, decay_rate=0, l2_lambda=0, use_avg=False):
        self.batch_size = batch_size
        self.beta = beta
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.use_avg = use_avg
        self.layer_to_moving_avg = collections.defaultdict(lambda: (0, 0))
        self.eps = 1e-8
        
    def optimize(trainable_layers):
        self.n_calls += 1
        if n_calls % self.batch_size:
            self.alpha = self.decay_rate * self.alpha
        
        for layer in trainable_layers:
            w = layer.w
            dw = layer.dw
            b = layer.b
            db = layer.db
            if self.use_avg:
                dw = np.divide(dw, self.batch_size)
                db = np.divide(db, self.batch_size)

            dw = dw + np.divide(self.decay_rate, self.batch_norm) * w
            
            w_avg, b_avg = self.layer_to_moving_avg[layer]
            w_new_avg, w = _calculateNewWeight(w, dw, w_avg)
            b_new_avg, b = _calculateNewWeight(b, db, b_avg)
            self.layer_to_moving_avg[layer] = (w_new_avg, b_new_avg)
            layer.update(new_dw, new_db)
            
        def _calculateNewWeight(x, dx, avg):
            new_avg = self.beta * avg + (1-self.beta)*np.square(dx)
            new_x = x - np.divide(self.alpha, new_avg + self.eps) * dx
            return new_avg, new_x