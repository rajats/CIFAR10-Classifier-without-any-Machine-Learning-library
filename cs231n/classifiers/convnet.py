import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet1(object):
    """
    A convolutional network with the following architecture:
  
    [conv-relu-pool]xL1 - conv - relu - [affine]xL2 - [softmax or SVM]
  
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self, hidden_dims, num_filters, filter_size, input_dim=(3, 32, 32), pool_size=2, 
               num_classes=10, conv_stride=1, pool_stride=2, num_conv_relu_pool=2, 
               num_affine=2, dropout=0, use_batchnorm=False, use_spbatchnorm=False, 
               weight_scale=1e-3, reg=0.0, dtype=np.float32, seed=None):
        """
        Initialize a new network.
    
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dims: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer
        - conv_stride: Stride for convolutional layer
        - pool_stride: Stride for pooling layer
        - num_conv_relu_pool: Number of conv-relu-pool layer
        - num_affine: Number of affine layer
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.conv_stride = conv_stride
        self.pool_stride = pool_stride
        self.num_conv_relu_pool = num_conv_relu_pool
        self.num_affine = num_affine
        self.use_dropout = dropout > 0
        self.use_batchnorm = use_batchnorm
        self.use_spbatchnorm = use_spbatchnorm
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim
        assert (self.filter_size - 1) % 2 == 0
        pad = (self.filter_size - 1) / 2
        H_conv_in = H
        W_conv_in = W
        depth = C
        for i in range(self.num_conv_relu_pool):
            self.params['W+%s' %(i+1)] = np.random.normal(0, weight_scale, (self.num_filters[i], depth, self.filter_size, self.filter_size))
            self.params['b+%s' %(i+1)] = np.zeros(self.num_filters[i])
            depth = self.num_filters[i]
            if self.use_spbatchnorm:
                self.params['gamma_sbn+%s' %(i+1)] = np.ones(depth)
                self.params['beta_sbn+%s' %(i+1)] = np.zeros(depth)
            assert (H_conv_in + 2 * pad - self.filter_size) % self.conv_stride == 0
            assert (W_conv_in + 2 * pad - self.filter_size) % self.conv_stride == 0
            H_conv_out = (H_conv_in + 2 * pad - self.filter_size) / self.conv_stride + 1
            W_conv_out = (W_conv_in + 2 * pad - self.filter_size) / self.conv_stride + 1
            assert (H_conv_out - self.pool_size) % self.pool_stride == 0
            assert (W_conv_out - self.pool_size) % self.pool_stride == 0
            H_pool_out = (H_conv_out - self.pool_size) / self.pool_stride + 1
            W_pool_out = (W_conv_out - self.pool_size) / self.pool_stride + 1
            H_conv_in = H_pool_out
            W_conv_in = W_pool_out
        var_idx = self.num_conv_relu_pool+1
        self.params['W+%s' %(var_idx)] = np.random.normal(0, weight_scale, (self.num_filters[var_idx-1], depth, self.filter_size, self.filter_size))
        self.params['b+%s' %(var_idx)] = np.zeros(self.num_filters[var_idx-1])
        depth = self.num_filters[var_idx-1]
        if self.use_spbatchnorm:
            self.params['gamma_sbn+%s' %(var_idx)] = np.ones(depth)
            self.params['beta_sbn+%s' %(var_idx)] = np.zeros(depth)
        assert (H_conv_in + 2 * pad - self.filter_size) % self.conv_stride == 0
        assert (W_conv_in + 2 * pad - self.filter_size) % self.conv_stride == 0
        H_conv_out = (H_conv_in + 2 * pad - self.filter_size) / self.conv_stride + 1
        W_conv_out = (W_conv_in + 2 * pad - self.filter_size) / self.conv_stride + 1
        input_dim = depth * W_conv_out * H_conv_out
        var_idx += 1
        if num_affine > 1:
            for i in range(self.num_affine):
                if i == 0:
                    self.params['W+%s' %(var_idx)] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[i]))
                    self.params['b+%s' %(var_idx)] = np.zeros(hidden_dims[i])
                elif i == self.num_affine - 1:
                    self.params['W+%s' %(var_idx)] = np.random.normal(0, weight_scale, (hidden_dims[i-1], num_classes))
                    self.params['b+%s' %(var_idx)] = np.zeros(num_classes)
                else:
                    self.params['W+%s' %(var_idx)] = np.random.normal(0, weight_scale, (hidden_dims[i-1], hidden_dims[i]))
                    self.params['b+%s' %(var_idx)] = np.zeros(hidden_dims[i])
                if use_batchnorm and i != (self.num_affine - 1):   # since outputs from the last layer of the network should not be normalized
                    self.params['gamma_bn+%s' %(i+1)] = np.ones(hidden_dims[i])
                    self.params['beta_bn+%s' %(i+1)] = np.zeros(hidden_dims[i])
                var_idx += 1
        else:
            self.params['W+%s' %(var_idx)] = np.random.normal(0, weight_scale, (input_dim, num_classes))
            self.params['b+%s' %(var_idx)] = np.zeros(num_classes)
            var_idx += 1
        self.sbn_params = []
        if self.use_spbatchnorm:
            self.sbn_params = [{'mode': 'train'} for i in xrange(self.num_conv_relu_pool+1)]

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_affine - 1)]
        
        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode 

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        if self.use_spbatchnorm:
            for sbn_param in self.sbn_params:
                sbn_param[mode] = mode

        conv_param = {'stride': self.conv_stride, 'pad': (self.filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': self.pool_size, 'pool_width': self.pool_size, 'stride': self.pool_stride}

        scores = None
        conv_spbatchnorm_relu_pool_cache = {}
        conv_relu_pool_cache = {}
        batchnorm_cache = {}
        affine_cache = {}
        dropout_cache = {}
        input_to_layer = X

        for i in range(self.num_conv_relu_pool):
            if self.use_spbatchnorm:
                conv_spbatchnorm_relu_pool_out, conv_spbatchnorm_relu_pool_cache[i] =  conv_spbatchnorm_relu_pool_forward(input_to_layer, 
                                                                                        self.params['W+%s' %(i+1)], self.params['b+%s' %(i+1)], 
                                                                                        conv_param, pool_param, self.params['gamma_sbn+%s' %(i+1)], 
                                                                                        self.params['beta_sbn+%s' %(i+1)], self.sbn_params[i])
                input_to_layer = conv_spbatchnorm_relu_pool_out
            else:
                conv_relu_pool_out, conv_relu_pool_cache[i] = conv_relu_pool_forward(input_to_layer, self.params['W+%s' %(i+1)], self.params['b+%s' %(i+1)], 
                                                                                    conv_param, pool_param)
                input_to_layer = conv_relu_pool_out

        var_idx = self.num_conv_relu_pool + 1
        if self.use_spbatchnorm:
            conv_spbatchnorm_relu_out, conv_spbatchnorm_relu_cache = conv_spbatchnorm_relu_forward(input_to_layer, self.params['W+%s' %(var_idx)], 
                                                                        self.params['b+%s' %(var_idx)], conv_param, self.params['gamma_sbn+%s' %(var_idx)], 
                                                                        self.params['beta_sbn+%s' %(var_idx)], self.sbn_params[var_idx-1])
            input_to_layer = conv_spbatchnorm_relu_out
        else:
            conv_relu_out, conv_relu_cache = conv_relu_forward(input_to_layer, self.params['W+%s' %(var_idx)], self.params['b+%s' %(var_idx)], conv_param)
            input_to_layer = conv_relu_out
        var_idx += 1
        for i in range(self.num_affine-1):
            out,  affine_cache[i] = affine_forward(input_to_layer, self.params['W+%s' %(var_idx)], self.params['b+%s' %(var_idx)])
            if self.use_batchnorm:
                out, batchnorm_cache[i] = batchnorm_forward(out, self.params['gamma_bn+%s' %(i+1)], self.params['beta_bn+%s' %(i+1)], 
                                                                    self.bn_params[i])
            input_to_layer = out
            if self.use_dropout:
                dropout_out, dropout_cache[i] = dropout_forward(input_to_layer, self.dropout_param)
                input_to_layer = dropout_out 
            var_idx += 1
        #last affine layer
        final_affine_layer_out,  final_affine_layer_cache = affine_forward(input_to_layer, self.params['W+%s' %(var_idx)], self.params['b+%s' %(var_idx)])
        scores = final_affine_layer_out

        if mode == 'test':
            return scores
        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y) 
        #last affine layer
        loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(var_idx)] ** 2)
        dx, dW, db = affine_backward(dscores, final_affine_layer_cache)
        grads['W+%s' %(var_idx)] = dW + (self.reg * self.params['W+%s' %(var_idx)])
        grads['b+%s' %(var_idx)] = db
        var_idx -= 1
        for i in reversed(range(self.num_affine-1)):
            if self.use_dropout:
                dx = dropout_backward(dx, dropout_cache[i])
            if self.use_batchnorm:
                dx, dgamma, dbeta =  batchnorm_backward(dx, batchnorm_cache[i])
                grads['gamma_bn+%s' %(i+1)] = dgamma
                grads['beta_bn+%s' %(i+1)] = dbeta 
            dx, dW, db = affine_backward(dx, affine_cache[i])
            grads['W+%s' %(var_idx)] = dW + (self.reg * self.params['W+%s' %(var_idx)])
            grads['b+%s' %(var_idx)] = db
            loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(var_idx)] ** 2)
            var_idx -= 1
        if self.use_spbatchnorm:
            dx, dW, db, dgamma, dbeta = conv_spbatchnorm_relu_backward(dx, conv_spbatchnorm_relu_cache)
            grads['gamma_sbn+%s' %(self.num_conv_relu_pool+1)] = dgamma
            grads['beta_sbn+%s' %(self.num_conv_relu_pool+1)] = dbeta 
        else:
            dx, dW, db = conv_relu_backward(dx, conv_relu_cache)
        grads['W+%s' %(var_idx)] = dW + (self.reg * self.params['W+%s' %(var_idx)])
        grads['b+%s' %(var_idx)] = db
        loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(var_idx)] ** 2)
        var_idx -= 1
        for i in reversed(range(self.num_conv_relu_pool)):
            if self.use_spbatchnorm:
                dx, dW, db, dgamma, dbeta = conv_spbatchnorm_relu_pool_backward(dx, conv_spbatchnorm_relu_pool_cache[i])
                grads['gamma_sbn+%s' %(i+1)] = dgamma
                grads['beta_sbn+%s' %(i+1)] = dbeta 
            else:
                dx, dW, db = conv_relu_pool_backward(dx, conv_relu_pool_cache[i])
            grads['W+%s' %(i+1)] = dW + (self.reg * self.params['W+%s' %(i+1)])
            grads['b+%s' %(i+1)] = db
            loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(i+1)] ** 2)

        return loss, grads

class ConvNet2(object):
    """
    A convolutional network with the following architecture:
  
    [conv-relu-conv-relu-pool]xL1 - [affine]xL2 - [softmax or SVM]
  
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self, hidden_dims, num_filters, filter_size, input_dim=(3, 32, 32), pool_size=2, 
               num_classes=10, conv_stride=1, pool_stride=2, num_conv_relu_x2_pool=2, 
               num_affine=2, dropout=0, use_batchnorm=False, use_spbatchnorm=False, 
               weight_scale=1e-3, reg=0.0, dtype=np.float32, seed=None):
        """
        Initialize a new network.
    
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer
        - conv_stride: Stride for convolutional layer
        - pool_stride: Stride for pooling layer
        - num_conv_relu_x2_pool: Number of conv-relu-conv-relu-pool layer
        - num_affine: Number of affine layer
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.conv_stride = conv_stride
        self.pool_stride = pool_stride
        self.num_conv_relu_x2_pool = num_conv_relu_x2_pool
        self.num_affine = num_affine
        self.use_dropout = dropout > 0
        self.use_batchnorm = use_batchnorm
        self.use_spbatchnorm = use_spbatchnorm
        self.reg = reg
        self.dtype = dtype
        C, H, W = input_dim
        assert (self.filter_size - 1) % 2 == 0
        pad = (self.filter_size - 1) / 2
        H_conv_in = H
        W_conv_in = W
        depth = C
        idx = 0
        for i in range(self.num_conv_relu_x2_pool):
            self.params['W+%s' %(idx+1)] = np.random.normal(0, weight_scale, (self.num_filters[idx], depth, self.filter_size, self.filter_size))
            self.params['b+%s' %(idx+1)] = np.zeros(self.num_filters[idx])
            depth = self.num_filters[idx]
            if self.use_spbatchnorm:
                self.params['gamma_sbn+%s' %(idx+1)] = np.ones(depth)
                self.params['beta_sbn+%s' %(idx+1)] = np.zeros(depth)
            assert (H_conv_in + 2 * pad - self.filter_size) % self.conv_stride == 0
            assert (W_conv_in + 2 * pad - self.filter_size) % self.conv_stride == 0
            H_conv_out = (H_conv_in + 2 * pad - self.filter_size) / self.conv_stride + 1
            W_conv_out = (W_conv_in + 2 * pad - self.filter_size) / self.conv_stride + 1
            idx += 1
            self.params['W+%s' %(idx+1)] = np.random.normal(0, weight_scale, (self.num_filters[idx], depth, self.filter_size, self.filter_size))
            self.params['b+%s' %(idx+1)] = np.zeros(self.num_filters[idx])
            depth = self.num_filters[idx]
            if self.use_spbatchnorm:
                self.params['gamma_sbn+%s' %(idx+1)] = np.ones(depth)
                self.params['beta_sbn+%s' %(idx+1)] = np.zeros(depth)
            H_conv_in = H_conv_out
            W_conv_in = W_conv_out
            assert (H_conv_in + 2 * pad - self.filter_size) % self.conv_stride == 0
            assert (W_conv_in + 2 * pad - self.filter_size) % self.conv_stride == 0
            H_conv_out = (H_conv_in + 2 * pad - self.filter_size) / self.conv_stride + 1
            W_conv_out = (W_conv_in + 2 * pad - self.filter_size) / self.conv_stride + 1
            idx += 1
            assert (H_conv_out - self.pool_size) % self.pool_stride == 0
            assert (W_conv_out - self.pool_size) % self.pool_stride == 0
            H_pool_out = (H_conv_out - self.pool_size) / self.pool_stride + 1
            W_pool_out = (W_conv_out - self.pool_size) / self.pool_stride + 1
            H_conv_in = H_pool_out
            W_conv_in = W_pool_out
        input_dim = depth * W_pool_out * H_pool_out
        if num_affine > 1:
            for i in range(self.num_affine):
                if i == 0:
                    self.params['W+%s' %(idx+1)] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[i]))
                    self.params['b+%s' %(idx+1)] = np.zeros(hidden_dims[i])
                elif i == self.num_affine - 1:
                    self.params['W+%s' %(idx+1)] = np.random.normal(0, weight_scale, (hidden_dims[i-1], num_classes))
                    self.params['b+%s' %(idx+1)] = np.zeros(num_classes)
                else:
                    self.params['W+%s' %(idx+1)] = np.random.normal(0, weight_scale, (hidden_dims[i-1], hidden_dims[i]))
                    self.params['b+%s' %(idx+1)] = np.zeros(hidden_dims[i])
                if use_batchnorm and i != (self.num_affine - 1):   # since outputs from the last layer of the network should not be normalized
                    self.params['gamma_bn+%s' %(i+1)] = np.ones(hidden_dims[i])
                    self.params['beta_bn+%s' %(i+1)] = np.zeros(hidden_dims[i])
                idx += 1
        else:
            self.params['W+%s' %(idx+1)] = np.random.normal(0, weight_scale, (input_dim, num_classes))
            self.params['b+%s' %(idx+1)] = np.zeros(num_classes)
            idx += 1
        self.sbn_params = []
        if self.use_spbatchnorm:
            self.sbn_params = [{'mode': 'train'} for i in xrange(self.num_conv_relu_x2_pool*2)]

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_affine - 1)]
        
        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode 

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        if self.use_spbatchnorm:
            for sbn_param in self.sbn_params:
                sbn_param[mode] = mode

        conv_param = {'stride': self.conv_stride, 'pad': (self.filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': self.pool_size, 'pool_width': self.pool_size, 'stride': self.pool_stride}

        scores = None
        conv_spbatchnorm_relu_cache = {}
        conv_relu_cache = {}
        batchnorm_cache = {}
        affine_cache = {}
        dropout_cache = {}
        pool_cache = {}
        input_to_layer = X
        idx = 0
        for i in range(self.num_conv_relu_x2_pool):
            if self.use_spbatchnorm:
                conv_spbatchnorm_relu_out, conv_spbatchnorm_relu_cache[idx] =  conv_spbatchnorm_relu_forward(input_to_layer, 
                                                                                        self.params['W+%s' %(idx+1)], self.params['b+%s' %(idx+1)], 
                                                                                        conv_param, self.params['gamma_sbn+%s' %(idx+1)], 
                                                                                        self.params['beta_sbn+%s' %(idx+1)], self.sbn_params[idx])
                input_to_layer = conv_spbatchnorm_relu_out
                idx += 1
                conv_spbatchnorm_relu_out, conv_spbatchnorm_relu_cache[idx] =  conv_spbatchnorm_relu_forward(input_to_layer, 
                                                                                        self.params['W+%s' %(idx+1)], self.params['b+%s' %(idx+1)], 
                                                                                        conv_param, self.params['gamma_sbn+%s' %(idx+1)], 
                                                                                        self.params['beta_sbn+%s' %(idx+1)], self.sbn_params[idx])
                input_to_layer = conv_spbatchnorm_relu_out
            else:
                conv_relu_out, conv_relu_cache[idx] = conv_relu_forward(input_to_layer, self.params['W+%s' %(idx+1)], self.params['b+%s' %(idx+1)], 
                                                                                    conv_param)
                input_to_layer = conv_relu_out
                idx += 1
                conv_relu_out, conv_relu_cache[idx] = conv_relu_forward(input_to_layer, self.params['W+%s' %(idx+1)], self.params['b+%s' %(idx+1)], 
                                                                                    conv_param)
                input_to_layer = conv_relu_out
            idx += 1
            pool_out, pool_cache[i] = max_pool_forward_fast(input_to_layer, pool_param)
            input_to_layer = pool_out

        for i in range(self.num_affine-1):
            out,  affine_cache[i] = affine_forward(input_to_layer, self.params['W+%s' %(idx+1)], self.params['b+%s' %(idx+1)])
            if self.use_batchnorm:
                out, batchnorm_cache[i] = batchnorm_forward(out, self.params['gamma_bn+%s' %(i+1)], self.params['beta_bn+%s' %(i+1)], 
                                                                    self.bn_params[i])
            input_to_layer = out
            if self.use_dropout:
                dropout_out, dropout_cache[i] = dropout_forward(input_to_layer, self.dropout_param)
                input_to_layer = dropout_out 
            idx += 1
        #last affine layer
        final_affine_layer_out,  final_affine_layer_cache = affine_forward(input_to_layer, self.params['W+%s' %(idx+1)], self.params['b+%s' %(idx+1)])
        scores = final_affine_layer_out

        if mode == 'test':
            return scores
        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y) 
        #last affine layer
        loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(idx+1)] ** 2)
        dx, dW, db = affine_backward(dscores, final_affine_layer_cache)
        grads['W+%s' %(idx+1)] = dW + (self.reg * self.params['W+%s' %(idx+1)])
        grads['b+%s' %(idx+1)] = db
        idx -= 1
        for i in reversed(range(self.num_affine-1)):
            if self.use_dropout:
                dx = dropout_backward(dx, dropout_cache[i])
            if self.use_batchnorm:
                dx, dgamma, dbeta =  batchnorm_backward(dx, batchnorm_cache[i])
                grads['gamma_bn+%s' %(i+1)] = dgamma
                grads['beta_bn+%s' %(i+1)] = dbeta 
            dx, dW, db = affine_backward(dx, affine_cache[i])
            grads['W+%s' %(idx+1)] = dW + (self.reg * self.params['W+%s' %(idx+1)])
            grads['b+%s' %(idx+1)] = db
            loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(idx+1)] ** 2)
            idx -= 1

        for i in reversed(range(self.num_conv_relu_x2_pool)):
            dx = max_pool_backward_fast(dx, pool_cache[i])
            if self.use_spbatchnorm:
                dx, dW, db, dgamma, dbeta = conv_spbatchnorm_relu_backward(dx, conv_spbatchnorm_relu_cache[idx])
                grads['gamma_sbn+%s' %(idx+1)] = dgamma
                grads['beta_sbn+%s' %(idx+1)] = dbeta 
                grads['W+%s' %(idx+1)] = dW + (self.reg * self.params['W+%s' %(idx+1)])
                grads['b+%s' %(idx+1)] = db
                loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(idx+1)] ** 2)
                idx -= 1
                dx, dW, db, dgamma, dbeta = conv_spbatchnorm_relu_backward(dx, conv_spbatchnorm_relu_cache[idx])
                grads['gamma_sbn+%s' %(idx+1)] = dgamma
                grads['beta_sbn+%s' %(idx+1)] = dbeta 
                grads['W+%s' %(idx+1)] = dW + (self.reg * self.params['W+%s' %(idx+1)])
                grads['b+%s' %(idx+1)] = db
                loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(idx+1)] ** 2)
            else:
                dx, dW, db = conv_relu_backward(dx, conv_relu_cache[idx])
                grads['W+%s' %(idx+1)] = dW + (self.reg * self.params['W+%s' %(idx+1)])
                grads['b+%s' %(idx+1)] = db
                loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(idx+1)] ** 2)
                idx -= 1
                dx, dW, db = conv_relu_backward(dx, conv_relu_cache[idx])
                grads['W+%s' %(idx+1)] = dW + (self.reg * self.params['W+%s' %(idx+1)])
                grads['b+%s' %(idx+1)] = db
                loss += 0.5 * self.reg * np.sum(self.params['W+%s' %(idx+1)] ** 2)
            idx -= 1
        return loss, grads





            
            
            