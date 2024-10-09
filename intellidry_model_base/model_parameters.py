#import pandas as pd
#import urllib.request
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import json
from pathlib import Path
from grain_params import grain_emc_params

"""
Class for holding, exporting and importing IntellyDry model parameters
"""
class ModelParameters:
  def __init__(self):
    # TODO: dataset_metadata should be a separate class
    self.dataset_name = '' # Name of the dataset the parameters are associated with
    self.folder_name = Path(__file__).resolve().parent / "datasets"
    self.constant_parameters = {} # Dictionary for storing model constants
    self.optimized_parameters = {} # Dictionary for storing optimized model parameters
    self.optimized_parameters_bounds = {} # Dictionary for storing bounds for optimized model parameters
    
  def set_const(self, key, value):
    self.constant_parameters[key] = value

  def set_opt(self, key, value, bounds):
    self.optimized_parameters[key] = value
    self.optimized_parameters_bounds[key] = bounds

  def load(self, dataset_name):
    self.dataset_name = dataset_name
    try:
      with open(f'{self.folder_name}/{dataset_name}_params.json', 'r') as f:
        data = json.load(f)
        # Update model parameters with loaded data
        self.constant_parameters.update(data['constant_parameters'])
        self.optimized_parameters.update(data['optimized_parameters'])
        self.optimized_parameters_bounds.update(data['optimized_parameters_bounds'])
        
        # Check grain type and set emc parameters accordingly
        try:
          grain_type = self.constant_parameters['grain_type']
          self.constant_parameters['grain_emc_params'] = grain_emc_params[grain_type]
        except KeyError:
          raise KeyError(f'Grain type "{grain_type}" not found in grain_emc_params')
    except Exception as e:
      raise Exception("Error loading model parameters: " + str(e))

  def save(self):
    try:
      with open(f'{self.folder_name}/{self.dataset_name}_params.json', 'w') as f:
        print(f'Saving model parameters to {self.folder_name}/{self.dataset_name}_params.json')
        json.dump({
          'constant_parameters': self.constant_parameters,
          'optimized_parameters': self.optimized_parameters,
          'optimized_parameters_bounds': self.optimized_parameters_bounds
        }, f, indent=4)
    except Exception as e:
      raise Exception("Error saving model parameters: " + str(e))

  def to_array(self):
    # recursively serialize constants considering "." as key separator for sub-dictionaries
    def serialize_dict(const_dict, prefix=''):
      serialized_dict = {}
      for key, value in const_dict.items():
        if isinstance(value, dict):
          serialized_dict.update(serialize_dict(value, prefix + key + '.'))
        else:
          serialized_dict[prefix + key] = value
      return serialized_dict
    
    return {
      'constant_parameters': serialize_dict(self.constant_parameters),
      'optimized_parameters': serialize_dict(self.optimized_parameters)
    }
  
  def update_optimized_parameters(self, param_array):
    # Iterate over param list of floats and sequentially update optimized parameters
    if len(param_array) != len(self.optimized_parameters):
      raise ValueError("Length of param_array does not match the number of optimized parameters")
    
    self.optimized_parameters.update(zip(self.optimized_parameters.keys(), param_array))
    
  
  
  # Add indexing operator to access constants and optimized_parameters
  def __getitem__(self, key):
    if key in self.constant_parameters:
      return self.constant_parameters[key]
    if key in self.optimized_parameters:
      return self.optimized_parameters[key]
    raise KeyError(f'Key "{key}" not found in model parameters')
  
  def print(self):
    print("Constant parameters:")
    print('\n'.join([f'{key}: {value}' for key, value in self.constant_parameters.items()]))
    print("Model parameters for dataset: " + self.dataset_name)
    print('\n'.join([f'{key}: {value}' for key, value in self.optimized_parameters.items()]))
    print("Optimized parameters:")
    print('\n'.join([f'{key}: {value}' for key, value in self.optimized_parameters_bounds.items()]))
