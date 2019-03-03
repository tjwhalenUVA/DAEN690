def setParameters(_dimensions,
                   _cnnFilters, _cnnKernel, _convActivation,
                   _cnnPool,
                   _cnnFlatten,
                   _cnnDense, _denseActivation,
                   _cnnDropout,
                   _outputActivation,
                   _summarize,
                   _optimizer):

    _dimensions=[10, 20]  # , 50, 100]
    _cnnFilters=[10]  # , 20, 50, 100]
    _cnnKernel=[2]  # , 5, 10]
    _cnnPool=[2]  # , 5, 10]
    _cnnFlatten=[True, False]
    _cnnDense=[10]  # , 20, 50]
    _cnnDropout=[None]  # , 0.10, 0.25, 0.50]

    return _dimensions, \
                   _cnnFilters, _cnnKernel, _convActivation, \
                   _cnnPool, \
                   _cnnFlatten, \
                   _cnnDense, _denseActivation, \
                   _cnnDropout, \
                   _outputActivation, \
                   _summarize, \
                   _optimizer
