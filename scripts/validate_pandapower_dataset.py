"""Certify stored labels against the final pandapower NR implementation."""
from genco_ptdf.validation import validate_dataset

if __name__ == "__main__":
    validate_dataset(["case14_ieee", "case118_ieee"])
