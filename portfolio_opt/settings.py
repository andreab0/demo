import logging

logger = logging.getLogger(__name__)

# Define a single place for big constants
BIG_M = 1e6    # previously was 1000 .... can alse set to 1e5, 1e9, etc. => a large number that suits your problem
EPSILON = 1e-12
