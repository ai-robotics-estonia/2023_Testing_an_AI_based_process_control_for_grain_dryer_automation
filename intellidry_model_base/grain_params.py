#!/usr/bin/env python3

# Empirical coefficients for equilibrium moisture content (EMC) Chung-Pfost model equations
# Source: Storage of Cereal Grains and Their Products, 5th Edition, page 272
grain_emc_params = {
    "barley": {"A": 457.12, "B": 0.14843, "C": 71.996},
    "buckwheat": {"A": 1.04E8, "B": 0.1646, "C": 1.59E7},
    "oats": {"A": 433.157, "B": 21.581, "C": 41.439},
    "rye": {"A": 461.023, "B": 0.184, "C": 38.741},
    "wheat": {"A": 610.34, "B": 0.15526, "C": 93.213},
    "peas": {"A": 433.157, "B": 21.581, "C": 41.439} # TODO: Add correct values for peas, currently using oats values
}
