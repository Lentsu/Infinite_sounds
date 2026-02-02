# freeze.py

A CLI freeze audio effect.

Velvet noise convolution method is based on [1].

FFT method based on [2].

## Installation

`pip install -r requirements.txt`

## Usage

```
python freeze.py <input.wav> <new duration [s]>
    (--output <output.wav>)
    (--method <freezing method>)
    (--window <window type id>)
    (--grain <grain-size [s]>)
    (--density <grain-density>)
    (--plot <plot_filepath.png>)
```

For more information, run `python freeze.py -h`.

## References

[1] S. D’Angelo and L. Gabrielli, “Efficient signal extrapolation by granulation and convolution with velvet noise”, in Proc. DAFX-18, Aveiro, Portugal, 2018

[2] V. Välimäki, J. Rämö, and F. Esqueda, “Creating endless sounds”, in Proc. DAFX-18, Aveiro, Portugal, 2018
