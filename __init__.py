
# /home/martin.feijoo/anaconda3/bin/python3


from . import core
from .core import *
from . import precip
from . import campos
from . import functions
__all__ = [core.__all__,
           'functions'
           'precip',
           'campos',
           ]



