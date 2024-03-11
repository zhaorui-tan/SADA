import tensorflow as tf
import numpy as np


# the code is modified from
# **Linear Algebra**

def symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.
    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.
    Also note that this method **only** works for symmetric matrices.
    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.
    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    s, u, v = tf.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)


def trace_sqrt_product(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root(sigma)
    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))
    return tf.linalg.trace(symmetric_matrix_square_root(sqrt_a_sigmav_a))


# **Estimators**
def sample_covariance(a, b, invert=False):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    assert (a.shape[0] == b.shape[0])
    assert (a.shape[1] == b.shape[1])
    m = a.shape[1]
    N = a.shape[0]
    C = tf.matmul(tf.transpose(a), b) / N
    if invert:
        return tf.linalg.pinv(C)
    else:
        return C

def sample_covariance_tune(a, b, invert=False):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    assert (a.shape[0] == b.shape[0])
    assert (a.shape[1] == b.shape[1])
    m = a.shape[1]
    N = a.shape[0]
    C = tf.matmul(tf.transpose(a), b) / N
    if invert:
        return tf.linalg.pinv(C, rcond=1)
    else:
        return C


def no_embedding(x):
    return x


# **Metrics**

# @tf.function
def ssd(y_true, y_predict, x_true, estimator=sample_covariance, calculate_TrSV=False):
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

    assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
    assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))

    # mean estimations
    m_y_true = tf.reduce_mean(y_true, axis=0)
    m_y_predict = tf.reduce_mean(y_predict, axis=0)
    m_x_true = tf.reduce_mean(x_true, axis=0)

    # to connect SC better and we have proved that cos(m_f,m_s) \propto E(e_f, e_s)
    # we use E[(e_f, es)] for our calculation
    # It will be the same if you use cos(m_f,m_s), only the value will be slightly different
    # SS = 1 - tf.reduce_sum(tf.math.multiply(m_y_predict, m_x_true))
    SS = 1 - tf.reduce_mean(tf.reduce_sum(tf.math.multiply(y_predict, x_true), axis=1))

    # covariance computations
    c_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)
    c_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)
    c_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
    c_x_true_y_predict = estimator(x_true - m_x_true, y_predict - m_y_predict)
    c_y_predict_y_predict = estimator(y_predict - m_y_predict, y_predict - m_y_predict)
    c_y_true_y_true = estimator(y_true - m_y_true, y_true - m_y_true)
    inv_c_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)

    c_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true)
    inv_c_y_true_y_true = estimator(y_true - m_y_true, x_true - m_y_true, invert=True)
    # conditoinal covariance estimations
    c_y_true_given_x_true = c_y_true_y_true - tf.matmul(c_y_true_x_true,
                                                        tf.matmul(inv_c_x_true_x_true, c_x_true_y_true))
    c_y_predict_given_x_true = c_y_predict_y_predict - tf.matmul(c_y_predict_x_true,
                                                                 tf.matmul(inv_c_x_true_x_true, c_x_true_y_predict))

    c_x_true_given_y_true = c_x_true_x_true - tf.matmul(c_x_true_y_true,
                                                        tf.matmul(inv_c_y_true_y_true, c_y_true_x_true))

    dSV = tf.math.sqrt(tf.math.square(tf.linalg.diag_part(c_y_predict_given_x_true - c_y_true_given_x_true)))
    dSV = tf.reduce_sum(dSV)

    mask = tf.cast(tf.linalg.diag(tf.ones([512])), dtype=tf.float64)
    if calculate_TrSV:
        с_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)

        с_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)
        с_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
        c_y_true_x_true_minus_c_y_predict_x_true = с_y_true_x_true - с_y_predict_x_true
        c_x_true_y_true_minus_c_x_true_y_predict = с_x_true_y_true - c_x_true_y_predict
        inv_с_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)

        m_dist = tf.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
        c_dist1 = tf.linalg.trace(tf.matmul(tf.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_с_x_true_x_true),
                                            c_x_true_y_true_minus_c_x_true_y_predict))
        TrSV = tf.linalg.trace(c_y_true_given_x_true * mask + c_y_predict_given_x_true * mask) - 2 * trace_sqrt_product(
            c_y_predict_given_x_true * mask, c_y_true_given_x_true * mask)
        return SS + dSV, SS, dSV, (m_dist, c_dist1, TrSV)
    return SS + dSV, SS, dSV, None

def ssd_save(y_true, y_predict, x_true, estimator=sample_covariance_tune, calculate_TrSV=False):
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

    assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
    assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))

    # mean estimations
    m_y_true = tf.reduce_mean(y_true, axis=0)
    m_y_predict = tf.reduce_mean(y_predict, axis=0)
    m_x_true = tf.reduce_mean(x_true, axis=0)

    # to connect SC better and we have proved that cos(m_f,m_s) \propto E(e_f, e_s)
    # we use E[(e_f, es)] for our calculation
    # It will be the same if you use cos(m_f,m_s), only the value will be slightly different
    # SS = 1 - tf.reduce_sum(tf.math.multiply(m_y_predict, m_x_true))
    SS = 1 - tf.reduce_mean(tf.reduce_sum(tf.math.multiply(y_predict, x_true), axis=1))

    # covariance computations
    c_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)
    c_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)
    c_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
    c_x_true_y_predict = estimator(x_true - m_x_true, y_predict - m_y_predict)
    c_y_predict_y_predict = estimator(y_predict - m_y_predict, y_predict - m_y_predict)
    c_y_true_y_true = estimator(y_true - m_y_true, y_true - m_y_true)
    inv_c_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)

    c_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true)
    inv_c_y_true_y_true = estimator(y_true - m_y_true, x_true - m_y_true, invert=True)
    # conditoinal covariance estimations
    c_y_true_given_x_true = c_y_true_y_true - tf.matmul(c_y_true_x_true,
                                                        tf.matmul(inv_c_x_true_x_true, c_x_true_y_true))
    c_y_predict_given_x_true = c_y_predict_y_predict - tf.matmul(c_y_predict_x_true,
                                                                 tf.matmul(inv_c_x_true_x_true, c_x_true_y_predict))

    c_x_true_given_y_true = c_x_true_x_true - tf.matmul(c_x_true_y_true,
                                                        tf.matmul(inv_c_y_true_y_true, c_y_true_x_true))
    # print(c_x_true_x_true)
    # print(c_x_true_y_true)
    # print(inv_c_y_true_y_true)
    # print(c_y_true_x_true)


    # a = c_x_true_x_true.numpy()
    # print(a)
    # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/cov/DAMSM/birds/c_image_true_image_true.npy", a)
    # a = c_x_true_given_y_true.numpy()
    # print(a)
    # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/cov/DAMSM/birds/c_image_true_given_text_true.npy", a)
    # np.save("/data1/phd21_zhaorui_tan/DF-GAN-master-new/birds/c_image_true_given_text_true.npy", a)
    # a = c_y_true_y_true.numpy()
    # print(a)
    # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/cov/DAMSM/birds/c_text_true_text_true.npy", a)
    # a = c_y_true_given_x_true.numpy()
    # print(a)
    # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/cov/DAMSM/birds/c_text_true_given_image_true.npy", a)
    # np.save("/data1/phd21_zhaorui_tan/DF-GAN-master-new/birds/c_text_true_given_image_true.npy", a)

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

    dSV = tf.math.sqrt(tf.math.square(tf.linalg.diag_part(c_y_predict_given_x_true - c_y_true_given_x_true)))
    dSV = tf.reduce_sum(dSV)

    mask = tf.cast(tf.linalg.diag(tf.ones([256])), dtype=tf.float64)
    if calculate_TrSV:
        с_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)

        с_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)
        с_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
        c_y_true_x_true_minus_c_y_predict_x_true = с_y_true_x_true - с_y_predict_x_true
        c_x_true_y_true_minus_c_x_true_y_predict = с_x_true_y_true - c_x_true_y_predict
        inv_с_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)

        m_dist = tf.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
        c_dist1 = tf.linalg.trace(tf.matmul(tf.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_с_x_true_x_true),
                                            c_x_true_y_true_minus_c_x_true_y_predict))
        TrSV = tf.linalg.trace(c_y_true_given_x_true * mask + c_y_predict_given_x_true * mask) - 2 * trace_sqrt_product(
            c_y_predict_given_x_true * mask, c_y_true_given_x_true * mask)
        return SS + dSV, SS, dSV, (m_dist, c_dist1, TrSV)
    return SS + dSV, SS, dSV, None

# # @tf.function
# def ssd(y_true, y_predict, x_true, estimator=sample_covariance, calculate_TrSV=False):
#     '''
#     y_true, y_predict, x_true should be normalized before calculating SSD
#     for SSD:
#         y_true: real image r
#         y_predict： fake image f
#         x_true： text es
#         ssd = 1 - cos(m_f,m_s)  + ||d(C_ff|s) - d(C_rr|s)||^2
#     for SSD-T:
#         y_true: real caption s
#         y_predict： fake caption sf
#         x_true： real image r
#         ssd = 1 - cos(m_fs,m_r)  + ||d(C_fsfs|s) - d(C_rr|s)||^2
#     '''
#
#     assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
#     assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))
#
#     # mean estimations
#     m_y_true = tf.reduce_mean(y_true, axis=0)
#     m_y_predict = tf.reduce_mean(y_predict, axis=0)
#     m_x_true = tf.reduce_mean(x_true, axis=0)
#
#     # to connect SC better and we have proved that cos(m_f,m_s) \propto E(e_f, e_s)
#     # we use E[(e_f, es)] for our calculation
#     # It will be the same if you use cos(m_f,m_s), only the value will be slightly different
#     # SS = 1 - tf.reduce_sum(tf.math.multiply(m_y_predict, m_x_true))
#     SS = 1 - tf.reduce_mean(tf.reduce_sum(tf.math.multiply(y_predict, x_true), axis=1))
#
#     # covariance computations
#     # c_y_predict_x_true = estimator((y_predict - m_y_predict) / tf.math.reduce_std(y_predict, 0),
#     #                                (x_true - m_x_true) / tf.math.reduce_std(x_true, 0)
#     #                                )
#     # c_y_true_x_true = estimator((y_true - m_y_true) / tf.math.reduce_std(y_true, 0),
#     #                             (x_true - m_x_true) / tf.math.reduce_std(x_true, 0)
#     #                             )
#     # c_x_true_y_true = estimator((x_true - m_x_true) / tf.math.reduce_std(x_true, 0),
#     #                             (y_true - m_y_true) / tf.math.reduce_std(y_true, 0)
#     #                             )
#     #
#     # c_x_true_y_predict = estimator((x_true - m_x_true) / tf.math.reduce_std(x_true, 0),
#     #                                (y_predict - m_y_predict) / tf.math.reduce_std(y_predict, 0)
#     #                                )
#     # c_y_predict_y_predict = estimator((y_predict - m_y_predict) / tf.math.reduce_std(y_predict, 0),
#     #                                   (y_predict - m_y_predict) / tf.math.reduce_std(y_predict, 0)
#     #                                   )
#     #
#     # c_y_true_y_true = estimator((y_true - m_y_true) / tf.math.reduce_std(y_true, 0),
#     #                             (y_true - m_y_true) / tf.math.reduce_std(y_true, 0)
#     #                             )
#     #
#     # inv_c_x_true_x_true = estimator((x_true - m_x_true) / tf.math.reduce_std(x_true, 0),
#     #                                 (x_true - m_x_true) / tf.math.reduce_std(x_true, 0),
#     #                                 invert=True)
#     #
#     # c_x_true_x_true = estimator((x_true - m_x_true) / tf.math.reduce_std(x_true, 0),
#     #                             (x_true - m_x_true) / tf.math.reduce_std(x_true, 0))
#     #
#     # inv_c_y_true_y_true = estimator((y_true - m_y_true) / tf.math.reduce_std(y_true, 0),
#     #                                 (x_true - m_y_true) / tf.math.reduce_std(x_true, 0), invert=True)
#
#     c_y_predict_x_true = estimator((y_predict - m_y_predict) / m_y_predict,
#                                    (x_true - m_x_true) / m_x_true
#                                    )
#     c_y_true_x_true = estimator((y_true - m_y_true) / m_y_true,
#                                 (x_true - m_x_true) / m_x_true
#                                 )
#     c_x_true_y_true = estimator((x_true - m_x_true) / m_x_true,
#                                 (y_true - m_y_true) / m_y_true
#                                 )
#
#     c_x_true_y_predict = estimator((x_true - m_x_true) / m_x_true,
#                                    (y_predict - m_y_predict) / m_y_predict
#                                    )
#     c_y_predict_y_predict = estimator((y_predict - m_y_predict) / m_y_predict,
#                                       (y_predict - m_y_predict) / m_y_predict
#                                       )
#
#     c_y_true_y_true = estimator((y_true - m_y_true) / m_y_true,
#                                 (y_true - m_y_true) / m_y_true
#                                 )
#
#     inv_c_x_true_x_true = estimator((x_true - m_x_true) / m_x_true,
#                                     (x_true - m_x_true) / m_x_true,
#                                     invert=True)
#
#     c_x_true_x_true = estimator((x_true - m_x_true) / m_x_true,
#                                 (x_true - m_x_true) / m_x_true
#                                 )
#
#     inv_c_y_true_y_true = estimator((y_true - m_y_true) / m_y_true,
#                                     (x_true - m_y_true) / m_y_true,
#                                     invert=True)
#     # conditoinal covariance estimations
#     c_y_true_given_x_true = c_y_true_y_true - tf.matmul(c_y_true_x_true,
#                                                         tf.matmul(inv_c_x_true_x_true, c_x_true_y_true))
#     c_y_predict_given_x_true = c_y_predict_y_predict - tf.matmul(c_y_predict_x_true,
#                                                                  tf.matmul(inv_c_x_true_x_true, c_x_true_y_predict))
#
#     c_x_true_given_y_true = c_x_true_x_true - tf.matmul(c_x_true_y_true,
#                                                         tf.matmul(inv_c_y_true_y_true, c_y_true_x_true))
#
#     a = c_x_true_x_true.numpy()
#     print(a)
#     np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/pc/DAMSM/birds/c_image_true_image_true.npy", a)
#     a = c_x_true_given_y_true.numpy()
#     print(a)
#     np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/pc/DAMSM/birds/c_image_true_given_text_true.npy", a)
#     a = c_y_true_y_true.numpy()
#     print(a)
#     np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/pc/DAMSM/birds/c_text_true_text_true.npy", a)
#     a = c_y_true_given_x_true.numpy()
#     print(a)
#     np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/pc/DAMSM/birds/c_text_true_given_image_true.npy", a)
#
#     # a = c_x_true_x_true.numpy()
#     # print(a)
#     # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/pc/DAMSM/coco/c_image_true_image_true.npy", a)
#     # a = c_x_true_given_y_true.numpy()
#     # print(a)
#     # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/pc/DAMSM/coco/c_image_true_given_text_true.npy", a)
#     # a = c_y_true_y_true.numpy()
#     # print(a)
#     # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/pc/DAMSM/coco/c_text_true_text_true.npy", a)
#     # a = c_y_true_given_x_true.numpy()
#     # print(a)
#     # np.save("/data1/phd21_zhaorui_tan/PDF_GAN_ssdloss/pc/DAMSM/coco/c_text_true_given_image_true.npy", a)
#
#     dSV = tf.math.sqrt(tf.math.square(tf.linalg.diag_part(c_y_predict_given_x_true - c_y_true_given_x_true)))
#     dSV = tf.reduce_sum(dSV)
#
#     mask = tf.cast(tf.linalg.diag(tf.ones([len(x_true[-1])])), dtype=tf.float64)
#     if calculate_TrSV:
#         с_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)
#
#         с_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)
#         с_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
#         c_y_true_x_true_minus_c_y_predict_x_true = с_y_true_x_true - с_y_predict_x_true
#         c_x_true_y_true_minus_c_x_true_y_predict = с_x_true_y_true - c_x_true_y_predict
#         inv_с_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)
#
#         m_dist = tf.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
#         c_dist1 = tf.linalg.trace(tf.matmul(tf.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_с_x_true_x_true),
#                                             c_x_true_y_true_minus_c_x_true_y_predict))
#         TrSV = tf.linalg.trace(c_y_true_given_x_true * mask + c_y_predict_given_x_true * mask) - 2 * trace_sqrt_product(
#             c_y_predict_given_x_true * mask, c_y_true_given_x_true * mask)
#         return SS + dSV, SS, dSV, (m_dist, c_dist1, TrSV)
#     return SS + dSV, SS, dSV, None


@tf.function
def cfid(y_true, y_predict, x_true, embeding=no_embedding, estimator=sample_covariance):
    '''
    CFID metric implementation, according to the formula described in the paper;
    https://arxiv.org/abs/2103.11521
    The formula:
    Given (x,y)~N(m_xy,C) and (x,y_h)~N(m_xy_h,C_h)
    Assume their joint Gaussian distribution:
    C = [[C_xx,   C_xy]
         [C_yx,   C_yy]]
    C_h = [[C_xx,   C_xy_h]
         [C_y_hx, C_y_hy_h]]
    m_xy = mean(x,y)
    m_xy_h = mean(x,y_h)
    Denote:
    C_y|x   = C_yy - C_yx @ (C_xx^-1) @ C_xy
    C_y_h|x = C_y_hy_h - C_y_hx @ (C_xx^-1) @ C_xy_h
    m_y     = mean(y)
    m_y_h   = mean(y_h)
    CFID((x,y), (x,y_h)) = ||m_y - m_y_h||^2 + Tr((C_yx-C_y_hx) @ (C_xx^-1) @ (C_xy-C_x_y_h)) + \
                                             + Tr(C_y|x + C_y_h|x) -2*Tr((C_y|x @ (C_y_h|x^(1/2)) @ C_y|x)^(1/2))
    The arguments:
    y_true    = [N,k1]
    y_predict = [N,k2]
    x_true    = [N,k3]
    embedding - Functon that transform [N,ki] -> [N,m], 'no_embedding' might be consider to used, if you working with same dimensions activations.
    estimator - Covariance estimator. Default is sample covariance estimator.
                The estimator might be switched to other estimators. Remmember that other estimator must support 'invert' argument
    '''

    y_predict = embeding(y_predict)
    y_true = embeding(y_true)
    x_true = embeding(x_true)

    assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
    assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))

    # mean estimations
    m_y_true = tf.reduce_mean(y_true, axis=0)
    m_y_predict = tf.reduce_mean(y_predict, axis=0)
    m_x_true = tf.reduce_mean(x_true, axis=0)

    # covariance computations
    с_y_predict_x_true = estimator(y_predict - m_y_predict, x_true - m_x_true)
    с_y_true_x_true = estimator(y_true - m_y_true, x_true - m_x_true)

    с_x_true_y_true = estimator(x_true - m_x_true, y_true - m_y_true)
    c_x_true_y_predict = estimator(x_true - m_x_true, y_predict - m_y_predict)

    с_y_predict_y_predict = estimator(y_predict - m_y_predict, y_predict - m_y_predict)
    с_y_true_y_true = estimator(y_true - m_y_true, y_true - m_y_true)
    inv_с_x_true_x_true = estimator(x_true - m_x_true, x_true - m_x_true, invert=True)

    # conditoinal mean and covariance estimations
    v = x_true - m_x_true
    A = tf.matmul(inv_с_x_true_x_true, tf.transpose(v))

    m_y_true_given_x_true = tf.reshape(m_y_true, (-1, 1)) + tf.matmul(с_y_true_x_true, A)
    m_y_predict_given_x_true = tf.reshape(m_y_predict, (-1, 1)) + tf.matmul(с_y_predict_x_true, A)

    c_y_true_given_x_true = с_y_true_y_true - tf.matmul(с_y_true_x_true,
                                                        tf.matmul(inv_с_x_true_x_true, с_x_true_y_true))
    c_y_predict_given_x_true = с_y_predict_y_predict - tf.matmul(с_y_predict_x_true,
                                                                 tf.matmul(inv_с_x_true_x_true, c_x_true_y_predict))
    c_y_true_x_true_minus_c_y_predict_x_true = с_y_true_x_true - с_y_predict_x_true
    c_x_true_y_true_minus_c_x_true_y_predict = с_x_true_y_true - c_x_true_y_predict

    # Distance between Gaussians
    m_dist = tf.einsum('...k,...k->...', m_y_true - m_y_predict, m_y_true - m_y_predict)
    c_dist1 = tf.linalg.trace(tf.matmul(tf.matmul(c_y_true_x_true_minus_c_y_predict_x_true, inv_с_x_true_x_true),
                                        c_x_true_y_true_minus_c_x_true_y_predict))
    c_dist2 = tf.linalg.trace(c_y_true_given_x_true + c_y_predict_given_x_true) - 2 * trace_sqrt_product(
        c_y_predict_given_x_true, c_y_true_given_x_true)

    return m_dist + c_dist1 + c_dist2



# def draw_distributions(save_dir, all_real, all_fake, all_caps, ):
#     import sklearn
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#     tsne = TSNE(n_components=2, init='pca', random_state=0)
#     print(all_real.shape)
#     data = np.stack((all_real, all_fake, all_caps, )).reshape(3*all_real.shape[0],all_real.shape[1] )
#     print(data.shape)
#     print(all_real,)
#     print(data)
#     data = sklearn.preprocessing.StandardScaler().fit_transform(data)
#     result = tsne.fit_transform(data)
#     x_min, x_max = result.min(0), result.max(0)
#     result = (result - x_min) / (x_max - x_min)
#
#     div = int(len(result) / 3)
#
#     for i in range(div - 1):
#         plt.scatter(result[i, 0], result[i, 1], marker='o',  # 此处的含义是用什么形状绘制当前的数据点
#                     color=plt.cm.Set1(1 / 10),  # 表示颜色
#                     s=20)  # 字体和大小
#         plt.scatter(result[div + i, 0], result[div + i, 1], marker='^',
#                     color=plt.cm.Set1(2 / 10),
#                     s=20)
#         plt.scatter(result[2 * div + i, 0], result[2 * div + i, 1], marker='+',
#                     color=plt.cm.Set1(3 / 10),
#                     s=50)
#     i = div - 1
#     plt.scatter(result[i, 0], result[i, 1], marker='o',  # 此处的含义是用什么形状绘制当前的数据点
#                 color=plt.cm.Set1(1 / 10),  # 表示颜色
#                 s=20, label='Real Image')  # 字体和大小
#     plt.scatter(result[div + i, 0], result[div + i, 1], marker='^',
#                 color=plt.cm.Set1(2 / 10),
#                 s=20, label='Fake Image')
#     plt.scatter(result[2 * div + i, 0], result[2 * div + i, 1], marker='+',
#                 color=plt.cm.Set1(3 / 10),
#                 s=50, label='Real Text')
#     plt.legend()
#     plt.savefig(f"{save_dir}/tsne_clip_distribution.pdf")
#     print(f'saved tsne_clip_distribution to {save_dir}/tsne_clip_distribution.pdf')