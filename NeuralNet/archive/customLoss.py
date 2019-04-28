
def customLoss(y_true, y_pred):
    import numpy as np
    from tensorflow import keras
    from . import backend as K

    thresholds = [-1.0e9, 0.5, 1.5, 2.5, 3.5, 1.0e9]

    yt = np.argmax(y_true, axis=1)
    yp = np.argmax(y_pred, axis=1)

    t1 = np.column_stack(list(((x+1e-6)-yp) for x in thresholds))
    t2 = np.column_stack(list((yp-(x-1e-6)) for x in thresholds))

    f1 = np.column_stack((yt >= x) for x in thresholds)
    f2 = np.column_stack((yt <= x) for x in thresholds)

    m1 = np.maximum(t1, np.zeros(shape=np.shape(t1)))
    m2 = np.maximum(t2, np.zeros(shape=np.shape(t2)))

    return K.sum(np.multiply(f1, m1) + np.multiply(f2, m2), axis=1)

