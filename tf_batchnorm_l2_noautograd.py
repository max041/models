import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

def _fused_batch_norm(x, scale, offset, mean, variance, epsilon=1e-5, data_format='NHWC', is_training=True, name=None):
    '''
    Batch normalization.

    Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
    The size of 1D Tensors matches the dimension C of the 4D Tensors.

    :param x: A 4D Tensor for input data.
    :param scale: A 1D Tensor for scaling factor, to scale the normalized x.
    :param offset: A 1D Tensor for offset, to shift to the normalized x.
    :param mean: A 1D Tensor for population mean. Used for inference only; must be empty for training.
    :param variance: A 1D Tensor for population variance. Used for inference only; must be empty for training.
    :param epsilon:  A small float number added to the variance of x. Defaults to `1e-5`.
    :param data_format: The data format for x and y. Either "NHWC" (default) or "NCHW".
    :param is_training: A bool value to indicate the operation is for training (default) or inference.
    :param name: A name for the operation (optional).
    :return: A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

      y: A 4D Tensor for output data.
      batch_mean: A 1D Tensor for mean.
      batch_variance: A 1D Tensor for variance.
      reserve_space_1: A 1D Tensor for mean.
      reserve_space_2: A 1D Tensor for inverse std.
    '''
    assert (data_format == 'NHWC') or (data_format == 'NCHW')

    @tf.custom_gradient
    def forward(x, running_mean, running_var, gamma, beta):
        with tf.name_scope(name if name else 'FusedBatchNorm'):
            if data_format == 'NHWC':
                x_ = tf.transpose(x, perm=[0, 3, 1, 2])  # N x C x H x W
            else:
                x_ = x  # N x C x H x W

            x_shape = tf.shape(x_)
            N, C, H, W = x_shape[0], x_shape[1], x_shape[2], x_shape[3]  # N x C x H x W
            y = tf.transpose(x_, perm=[1, 0, 2, 3])  # C x N x H x W
            y = tf.reshape(y, shape=[C, N * H * W])  # C x (N*H*W)
            N_rcp = 1 / tf.cast(tf.shape(y)[1], dtype=tf.float32)

            if is_training:
                mean = tf.reduce_sum(y, axis=1)  # C
                mean = mean * N_rcp

                xmean = y - tf.expand_dims(mean, -1)

                # Whole variance calculation can be done in fp32, w/o performance penalty. This is supported by MPU according to Maciej.
                xvar = xmean ** 2
                xvar = tf.reduce_sum(xvar, axis=1)  # C
                xvar = N_rcp * xvar

                # According to Maciej result of sum can be directly sent to rsqrt lut in fp32 w/o performance penalty
                scale = tf.rsqrt(xvar + epsilon)
            else:
                mean = running_mean
                xvar = running_var
                scale = tf.rsqrt(xvar + epsilon)

            xmean = x_ - tf.reshape(mean, shape=[1, C, 1, 1])
            xhat = tf.reshape(scale, shape=[1, C, 1, 1]) * xmean

            out = xhat * tf.reshape(gamma, shape=[1, C, 1, 1])
            out = out + tf.reshape(beta, shape=[1, C, 1, 1])

            if data_format == 'NHWC':
                out = tf.transpose(out, perm=[0, 2, 3, 1])  # N x H x W x C

        def backward(grad_output, grad_mean, grad_xvar, grad_mean_2, grad_scale):
            with tf.name_scope(name + 'Grad' if name else 'FusedBatchNormGrad'):
                if data_format == 'NHWC':
                    grad_output_ = tf.transpose(grad_output, perm=[0, 3, 1, 2])  # N x C x H x W
                else:
                    grad_output_ = grad_output

                grad_output_shape = tf.shape(grad_output_)
                N, C, H, W = grad_output_shape[0], grad_output_shape[1], grad_output_shape[2], grad_output_shape[3]  # N x C x H x W
                y = tf.transpose(grad_output_, perm=[1, 0, 2, 3])  # C x N x H x W
                y = tf.reshape(y, shape=[C, N * H * W])  # C x (N*H*W)

                N_rcp = 1 / tf.cast(tf.shape(y)[1], dtype=tf.float32)
                grad_gamma = tf.reshape(tf.transpose(xhat, perm=[1, 0, 2, 3]), shape=[C, N * H * W]) * y
                grad_gamma = tf.reduce_sum(grad_gamma, axis=1)
                grad_beta = tf.reduce_sum(y, axis=1)

                grad = xhat * tf.reshape(grad_gamma, shape=[1, C, 1, 1])
                grad = N_rcp * (grad + tf.reshape(grad_beta, shape=[1, C, 1, 1]))
                grad = grad_output_ - grad
                grad = grad * tf.reshape(scale, shape=[1, C, 1, 1]) * tf.reshape(gamma, shape=[1, C, 1, 1])

                if data_format == 'NHWC':
                    grad = tf.transpose(grad, perm=[0, 2, 3, 1])  # N x H x W x C

            return grad, tf.constant([]), tf.constant([]), grad_gamma, grad_beta

        return (out, mean, xvar, mean, scale), backward

    y, batch_mean, batch_variance, reserve_space_1, reserve_space_2 = forward(x, mean, variance, scale, offset)

    return y, batch_mean, batch_variance, reserve_space_1, reserve_space_2


if __name__ == '__main__':
    import numpy as np
    # init data ...
    x_np = np.random.rand(64, 3, 32, 32).astype(np.float32)  # N C H W
    scale_np = np.random.rand(3).astype(np.float32)  # C
    offset_np = np.random.rand(3).astype(np.float32)  # C
    mean_np = np.array([]).astype(np.float32)  # empty
    variance_np = np.array([]).astype(np.float32)  # empty
    eps_np = np.ones((64, 3, 32, 32)).astype(np.float32)

    # create TF custom batch normalization ...
    x = tf.placeholder(tf.float32, name='x')
    scale = tf.placeholder(tf.float32, name='scale')
    offset = tf.placeholder(tf.float32, name='offset')
    mean = tf.placeholder(tf.float32, name='mean')
    variance = tf.placeholder(tf.float32, name='variance')

    y, batch_mean, batch_variance, reserve_space_1, reserve_space_2 = _fused_batch_norm(
        x, scale, offset, mean, variance, data_format='NCHW')

    fetches = [y, batch_mean, batch_variance, reserve_space_1, reserve_space_2]
    feed_dict = {
        x: x_np,
        scale: scale_np,
        offset: offset_np,
        mean: mean_np,
        variance: variance_np
    }

    with tf.Session() as sess:
        outputs = sess.run(fetches=y, feed_dict=feed_dict)

    print(outputs)