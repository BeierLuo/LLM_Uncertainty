import warnings
from .haloscope import HaloScope

def get_estimator(args, dataset, index_dict):
    if args.estimator_name == 'haloscope':
        return HaloScope(args, dataset, index_dict)
    else:
        warnings.warn(f"Warning: The method '{args.estimator_name}' is not implemented yet.", UserWarning)
        return None