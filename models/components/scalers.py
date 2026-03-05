from copy import deepcopy

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "Scaler",
    "StandardScaler",
    "MinMaxScaler",
]


def zeros_to_one_(scale):
    """Set to 1 scales of near constant features, detected by identifying
    scales close to machine precision, in place.
    Adapted from :class:`sklearn.preprocessing._data._handle_zeros_in_scale`
    """
    if np.isscalar(scale):
        return 1.0 if np.isclose(scale, 0.0) else scale
    eps = 10 * np.finfo(scale.dtype).eps
    zeros = np.isclose(scale, 0.0, atol=eps, rtol=eps)
    scale[zeros] = 1.0
    return scale


def fit_wrapper(fit_function):
    def fit(obj: "Scaler", x, *args, **kwargs) -> "Scaler":
        x_type = type(x)
        x = np.asarray(x)
        fit_function(obj, x, *args, **kwargs)
        if x_type is Tensor:
            obj.torch()
        return obj

    return fit


class Scaler:
    r"""Base class for linear :class:`~tsl.data.SpatioTemporalDataset` scalers.

    A :class:`~tsl.data.preprocessing.Scaler` is the base class for
    linear scaler objects. A linear scaler apply a linear transformation to the
    input using parameters `bias` :math:`\mu` and `scale` :math:`\sigma`:

    .. math::
      f(x) = (x - \mu) / \sigma.

    Args:
        bias (float): the offset of the linear transformation.
            (default: 0.)
        scale (float): the scale of the linear transformation.
            (default: 1.)
    """

    def __init__(self, bias=0.0, scale=1.0):
        self.bias = bias
        self.scale = scale
        super(Scaler, self).__init__()

    def __repr__(self) -> str:
        sizes = []
        for k, v in self.params().items():
            param = f"{k}={tuple(v.shape) if hasattr(v, 'shape') else v}"
            sizes.append(param)
        return "{}({})".format(self.__class__.__name__, ", ".join(sizes))

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def params(self) -> dict:
        """Dictionary of the scaler parameters `bias` and `scale`.

        Returns:
            dict: Scaler's parameters `bias` and `scale.`
        """
        return dict(bias=self.bias, scale=self.scale)

    def torch(self, inplace=True):
        scaler = self
        if not inplace:
            scaler = deepcopy(self)
        for name, param in scaler.params().items():
            param = torch.atleast_1d(torch.as_tensor(param))
            setattr(scaler, name, param)
        return scaler

    def numpy(self, inplace=True):
        r"""Transform all tensors to numpy arrays."""
        scaler = self
        if not inplace:
            scaler = deepcopy(self)
        for name, param in scaler.params().items():
            if isinstance(param, Tensor):
                param = param.detach().cpu().numpy()
            setattr(scaler, name, param)
        return scaler

    @fit_wrapper
    def fit(self, x, *args, **kwargs):
        """Fit scaler's parameters using input :obj:`x`."""
        raise NotImplementedError()

    def transform(self, x):
        return (x - self.bias) / self.scale + 5e-8

    def inverse_transform(self, x):
        return (x - 5e-8) * self.scale + self.bias

    def fit_transform(self, x, *args, **kwargs):
        """Fit scaler's parameters using input :obj:`x` and then transform
        :obj:`x`."""
        self.fit(x, *args, **kwargs)
        return self.transform(x)


class StandardScaler(Scaler):
    """Apply standardization to data by removing mean and scaling to unit
    variance.

    Args:
        axis (int): dimensions of input to fit parameters on.
            (default: 0)
    """

    def __init__(self, axis=0):
        super(StandardScaler, self).__init__()
        self.axis = axis

    @fit_wrapper
    def fit(self, x, mask=None, keepdims=True):
        if mask is not None:
            x = np.where(mask, x, np.nan)
            self.bias = np.nanmean(
                x.astype(np.float32), axis=self.axis, keepdims=keepdims
            ).astype(x.dtype)
            self.scale = np.nanstd(
                x.astype(np.float32), axis=self.axis, keepdims=keepdims
            ).astype(x.dtype)
        else:
            self.bias = x.mean(axis=self.axis, keepdims=keepdims)
            self.scale = x.std(axis=self.axis, keepdims=keepdims)
        self.scale = zeros_to_one_(self.scale)
        return self


class MinMaxScaler(Scaler):
    """Rescale data such that all lay in the specified range (default is
    :math:`[0,1]`).

    Args:
        axis (int): dimensions of input to fit parameters on.
            (default: 0)
        out_range (tuple): output range of transformed data.
            (default: :obj:`(0, 1)`)
    """

    def __init__(self, axis=0, out_range=(0.0, 1.0)):
        super(MinMaxScaler, self).__init__()
        self.axis = axis
        self.out_range = out_range

    @fit_wrapper
    def fit(self, x, mask=None, keepdims=True):

        out_min, out_max = self.out_range
        if out_min >= out_max:
            raise ValueError(
                "Output range minimum must be smaller than maximum. Got {}.".format(
                    self.out_range
                )
            )

        if mask is not None:
            x = np.where(mask, x, np.nan)
            x_min = np.nanmin(
                x.astype(np.float32), axis=self.axis, keepdims=keepdims
            ).astype(x.dtype)
            x_max = np.nanmax(
                x.astype(np.float32), axis=self.axis, keepdims=keepdims
            ).astype(x.dtype)
        else:
            x_min = x.min(axis=self.axis, keepdims=keepdims)
            x_max = x.max(axis=self.axis, keepdims=keepdims)
        scale = (x_max - x_min) / (out_max - out_min)
        scale = zeros_to_one_(scale)
        bias = x_min - out_min * scale
        self.bias, self.scale = bias, scale
        return self
