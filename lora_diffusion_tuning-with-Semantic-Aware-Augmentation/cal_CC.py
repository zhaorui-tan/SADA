import numpy as np
# import tensorflow as tf
import torch

# **Estimators**
def sample_covariance(a, b, invert=False):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    # assert (a.shape[0] == b.shape[0])
    # assert (a.shape[1] == b.shape[1])
    # m = a.shape[1]
    N = a.shape[0]
    C = torch.matmul(a.T, b)/ N
    if invert:
        return torch.linalg.pinv(C,)
    else:
        return C

# def sample_covariance_tune(a, b, invert=False):
#     '''
#     Sample covariance estimating
#     a = [N,m]
#     b = [N,m]
#     '''
#     assert (a.shape[0] == b.shape[0])
#     assert (a.shape[1] == b.shape[1])
#     m = a.shape[1]
#     N = a.shape[0]
#     C = tf.matmul(tf.transpose(a), b) / N
#     if invert:
#         return tf.linalg.pinv(C, rcond=1)
#     else:
#         return C


def no_embedding(x):
    return x


# **Metrics**
def ssd_save(y_true,  x_true, estimator=sample_covariance, calculate_TrSV=False):
    '''
    y_true, y_predict, x_true should be normalized before calculating SSD
    for SSD:
        y_true: real image r
        y_predict： fake image f
        x_true： text es
        ssd = 1 - cos(m_f,m_s)  + ||d(C_ff|s) - d(C_rr|s)||^2
    for SSD-T:
        y_true: real caption s
        y_predict： fake caption sf
        x_true： real image r
        ssd = 1 - cos(m_fs,m_r)  + ||d(C_fsfs|s) - d(C_rr|s)||^2
    '''

    # assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
    # assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))

    # mean estimations
    m_y_true = torch.mean(y_true, dim=0)

    m_x_true = torch.mean(x_true, dim=0)

    # to connect SC better and we have proved that cos(m_f,m_s) \propto E(e_f, e_s)
    # we use E[(e_f, es)] for our calculation
    # It will be the same if you use cos(m_f,m_s), only the value will be slightly different
    # SS = 1 - tf.reduce_sum(tf.math.multiply(m_y_predict, m_x_true))

    # covariance computations
    # c_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)
    c_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)
    c_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)

    c_y_true_y_true = estimator(y_true - m_y_true, y_true - m_y_true)
    inv_c_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)

    c_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true)
    inv_c_y_true_y_true = estimator(y_true - m_y_true, y_true - m_y_true, invert=True)
    # conditoinal covariance estimations
    c_y_true_given_x_true = c_y_true_y_true - torch.matmul(c_y_true_x_true,
                                                        torch.matmul(inv_c_x_true_x_true, c_x_true_y_true))

    c_x_true_given_y_true = c_x_true_x_true - torch.matmul(c_x_true_y_true,
                                                        torch.matmul(inv_c_y_true_y_true, c_y_true_x_true))
    print(c_x_true_x_true)
    print(c_x_true_y_true)
    print(inv_c_y_true_y_true)
    print(c_y_true_x_true)

    # np.save("c_image_true_image_true.npy", a)
    a = c_x_true_given_y_true.numpy()
    print(a)
    print(a.shape)
    np.save("c_image_true_given_text_true.npy", a)
    a = c_y_true_given_x_true.numpy()
    print(a)
    print(a.shape)
    np.save("c_text_true_given_image_true.npy", a)

    # a = c_x_true_x_true.numpy()
    # print(a)
    # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/cov/DAMSM/coco/c_image_true_image_true.npy", a)
    # a = c_x_true_given_y_true.numpy()
    # print(a)
    # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/cov/DAMSM/coco/c_image_true_given_text_true.npy", a)
    # a = c_y_true_y_true.numpy()
    # print(a)
    # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/cov/DAMSM/coco/c_text_true_text_true.npy", a)
    # a = c_y_true_given_x_true.numpy()
    # print(a)
    # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/cov/DAMSM/coco/c_text_true_given_image_true.npy", a)

    # dSV = tf.math.sqrt(tf.math.square(tf.linalg.diag_part(c_y_predict_given_x_true - c_y_true_given_x_true)))
    # dSV = tf.reduce_sum(dSV)
    #
    # mask = tf.cast(tf.linalg.diag(tf.ones([256])), dtype=tf.float64)
    # if calculate_TrSV:
    #     с_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)
    #
    #     с_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)
    #     с_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
    #     c_y_true_x_true_minus_c_y_predict_x_true = с_y_true_x_true - с_y_predict_x_true
    #     c_x_true_y_true_minus_c_x_true_y_predict = с_x_true_y_true - c_x_true_y_predict
    #     inv_с_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)
    #
    #     m_dist = tf.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
    #     c_dist1 = tf.linalg.trace(tf.matmul(tf.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_с_x_true_x_true),
    #                                         c_x_true_y_true_minus_c_x_true_y_predict))
    #     TrSV = tf.linalg.trace(c_y_true_given_x_true * mask + c_y_predict_given_x_true * mask) - 2 * trace_sqrt_product(
    #         c_y_predict_given_x_true * mask, c_y_true_given_x_true * mask)
    #     return SS + dSV, SS, dSV, (m_dist, c_dist1, TrSV)
    # return SS + dSV, SS, dSV, None


if __name__ == '__main__':
    hiddens_text = np.load('clip_hidden.npy')
    hiddens_image = np.load('clip_hidden_images.npy')


    hiddens_text = torch.tensor(hiddens_text, dtype=torch.float32).mean(1)
    hiddens_image = torch.tensor(hiddens_image, dtype=torch.float32)
    print(hiddens_text.shape)
    print(hiddens_image.shape)
    hiddens_text_shape = hiddens_text.shape
    # hiddens_text = hiddens_text.view(hiddens_text_shape[0],hiddens_text_shape[1]*hiddens_text_shape[2])
    # hiddens_text = hiddens_text/tf.linalg.normalize(hiddens_text)
    ssd_save(hiddens_text, hiddens_image)