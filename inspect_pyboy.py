import inspect
from pyboy import PyBoy

try:
    sig = inspect.signature(PyBoy.__init__)
    print(sig)
except Exception as e:
    print(e)

print(PyBoy.__init__.__doc__)
