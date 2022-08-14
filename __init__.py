from _spacemap import SpaceMAP

import numba

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("spacemap").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.5-dev"
