import pandas as pd
import numpy as np
import os
from pathlib import Path
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

"""
Dataset class for storing, loading and preprocessing IntellyDry datasets
"""

class Dataset:
  def __init__(self):
    self.dataset_name = ''              # Name of the dataset
    self.dataset_folder = Path(__file__).resolve().parent / "datasets"
    self.grain_type = ''                # Type of grain in the dataset
    self.df = pd.DataFrame()            # Pandas dataframe for preprocessed (filtered, interpolated, etc) dataset
    self.df_raw = pd.DataFrame()        # Pandas dataframe for storing original unprocessed dataset
    self.start_time = None              # Start timestamp of processed dataset
    self.end_time = None                # End timestamp of processed dataset
    self.dt = pd.Timedelta(seconds=10)  # Resampling interval as Timedelta object
    
  def load_example_dataset(self, dataset_name, FORCE_DOWNLOAD = False, start_time = None, end_time = None, autodetect_drying_interval = False):
    self.dataset_name = dataset_name
    self.start_time = start_time
    self.end_time = end_time
    self.grain_type = self.get_grain_type()

    print("Loading dataset: " + dataset_name)
    print("Grain type identified as: " + self.grain_type)
    
    # Check if the dataset directory exists
    if not os.path.exists(self.dataset_folder):
      print (f"Datasets directory not found. Creating datasets directory: {self.dataset_folder}")
      os.makedirs(self.dataset_folder)
    
    # Prepare dataset_local_path
    dataset_local_path = f'{self.dataset_folder}/{dataset_name}.csv'

    # Check if the dataset file exists
    if not os.path.exists(f'{dataset_local_path}') or FORCE_DOWNLOAD:
      # Download the dataset from GitHub
      dataset_remote_path = f'https://raw.githubusercontent.com/intellidry/intellidry-public-datasets/main/{dataset_name}/merged_data.csv'
      print("Downloading dataset from " + dataset_remote_path)
      try:
        urllib.request.urlretrieve(dataset_remote_path, dataset_local_path)
      except Exception as e:
        raise Exception("Error downloading dataset: " + str(e))
    else:
      print("Using local dataset file: " + dataset_local_path)

    # Load the dataset into a pandas dataframe
    try:
      self.df = pd.read_csv(dataset_local_path)
    except Exception as e:
      raise Exception("Error loading dataset: " + str(e))
   
    # Column name remap for example datasets
    column_name_remap = {
      'temp1': 'temp_heated',
      'temp2': 'temp_out',
      'temp3': 'temp_in',
      'temp4': 'temp_low',
      'temp5': 'temp_high',
      'temp6': 'temp_mid',
      'rh_1': 'rh_in',
      'rh_2': 'rh_out'
    }

    # Drop the first index column, rename the columns and set the datetime
    self.df = self.df.drop(self.df.columns[0], axis=1)
    self.df = self.df.rename(columns=column_name_remap)

    # Check for start and end times and initialize to entire dataset if not provided
    if self.start_time is None:
      self.start_time = self.df.index[0]
    if self.end_time is None:
      self.end_time = self.df.index[-1]
    
    # If timestamps are strings, convert to a pandas datetime objects
    if isinstance(self.start_time, str):
      self.start_time = pd.Timestamp(self.start_time)
    if isinstance(self.end_time, str):
      self.end_time = pd.Timestamp(self.end_time)

    print("Preprocessing dataset: %s, period: %s - %s" % (self.dataset_name, self.start_time.strftime("%H:%M"), self.end_time.strftime("%H:%M")))
   
    try:
      if self.dataset_name == 'buckwheat_2023_10_19': # Special case for buckwheat dataset formatting
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], format='%H:%M:%S')
        self.df['datetime'] = self.df['datetime'].apply(lambda x: x.replace(year=2023, month=10, day=19))
      else:
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], format='%d/%m/%Y %H:%M')
      self.df = self.df.set_index('datetime')

      # Create a snapshot for the raw dataset
      self.df_raw = self.df.copy()

      # Filter the dataset based on the start and end times
      self.df = self.df.loc[self.start_time:self.end_time]

      # Resample the data to 10-second intervals and interpolate missing values
      self.df = self.df.resample(self.dt).mean().interpolate(method='linear', limit_direction='both')

    except Exception as e:
      raise Exception("Error pre-processing dataframe: " + dataset_name + ": " + str(e))
    
        
    # Narrow the dataset to the autodetected drying interval
    if autodetect_drying_interval:
      self.start_time, self.end_time = self.detect_drying_interval()
      self.df = self.df.loc[self.start_time:self.end_time]

  # TODO: Until no database is connected, this function is used to determine the grain type from the dataset name
  def get_grain_type(self):
    if 'barley' in self.dataset_name:
      return 'barley'
    elif 'buckwheat' in self.dataset_name:
      return 'buckwheat'
    elif 'oats' in self.dataset_name:
      return 'oats'
    elif 'rye' in self.dataset_name:
      return 'rye'
    elif 'wheat' in self.dataset_name:
      return 'wheat'
    elif 'peas' in self.dataset_name:
      return 'peas'
    else:
      return 'unknown'  
  
  def plot(self, raw=False, drier_temps_only=False):
    df = self.df_raw if raw else self.df

    # Visualize the dataframe
    if not drier_temps_only:
      plt.plot(df.index, df["moist"], label="measured grain moisture", linestyle="dotted")
      plt.plot(df.index, df["rh_in"], label="rh_in", linestyle="dotted")
      plt.plot(df.index, df["rh_out"], label="rh_out", linestyle="dotted")
      plt.plot(df.index, df["temp_in"], label="temp_in")
      plt.plot(df.index, df["temp_heated"], label="temp_heated")
      plt.plot(df.index, df["temp_out"], label="temp_out")
    plt.plot(df.index, df["temp_mid"], label="temp_mid")
    plt.plot(df.index, df["temp_low"], label="temp_low")
    plt.plot(df.index, df["temp_high"], label="temp_high")
       
    # Draw vertical cursors for the beginning end end of the df if raw data is plotted
    if raw:
      plt.axvline(x=self.start_time, color='gray', linestyle='--', linewidth=1, label='start time')
      plt.axvline(x=self.end_time, color='gray', linestyle='--', linewidth=1, label='end time')

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    #make grid
    plt.grid(True)

    #set x axis  15 min
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))

    plt.ylabel("degrees in C and RH%")
    plt.title(self.dataset_name)
    plt.legend(loc='upper center', bbox_to_anchor=(1.02, 1.03))
    plt.show(block=False)
    return plt.gca()


  '''
  Methods for 2024 data.
  Only reads from local data.
  Does not interpolate 
  '''
  def load_example_dataset_2024(self, dataset_name, FORCE_DOWNLOAD = False, start_time = None, end_time = None, autodetect_drying_interval = False):
    dataset_name = dataset_name
    self.dataset_name = dataset_name
    self.start_time = start_time
    self.end_time = end_time
    self.grain_type = self.get_grain_type()

    print ("Loading dataset: " + self.dataset_folder+"/"+dataset_name)
    print("Grain type identified as: " + self.grain_type)
    
    # Check if the dataset directory exists
    if not os.path.exists(self.dataset_folder):
      print ("Datasets directory not found. Creating datasets directory.")
      os.makedirs(self.dataset_folder)
    
    # Prepare dataset_local_path
    dataset_local_path = f'{self.dataset_folder}/{dataset_name}.csv'

    # Load the dataset into a pandas dataframe
    try:
      self.df = pd.read_csv(dataset_local_path)
    except Exception as e:
      raise Exception("Error loading dataset: " + str(e))
   
    # Column name remap for example datasets
    column_name_remap = {
      'temp1': 'temp_heated',
      'temp2': 'temp_out',
      'temp3': 'temp_in',
      'temp4': 'temp_low',
      'temp5': 'temp_high',
    }

    # Drop the first index column, rename the columns and set the datetime
    self.df = self.df.drop(self.df.columns[0], axis=1)
    self.df = self.df.rename(columns=column_name_remap)
    # Drop the first index column, rename the columns and set the datetime
    self.df = self.df.drop(self.df.columns[0], axis=1)
 
    # Check for start and end times and initialize to entire dataset if not provided
    if self.start_time is None:
      self.start_time = self.df['datetime'][ self.df.index[0]]
    if self.end_time is None:
      self.end_time = self.df['datetime'][ self.df.index[-1]]

    # If timestamps are strings, convert to a pandas datetime objects
    if isinstance(self.start_time, str):
      self.start_time = pd.Timestamp(self.start_time)
    if isinstance(self.end_time, str):
      self.end_time = pd.Timestamp(self.end_time)

    print("Preprocessing dataset: %s, period: %s - %s" % (self.dataset_name, self.start_time.strftime("%H:%M"), self.end_time.strftime("%H:%M")))
    try:
      if self.dataset_name == 'buckwheat_2023_10_19': # Special case for buckwheat dataset formatting
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], format='%H:%M:%S')
        self.df['datetime'] = self.df['datetime'].apply(lambda x: x.replace(year=2023, month=10, day=19))
      else:
        #self.df['datetime'] = pd.to_datetime(self.df['datetime'], format='%d/%m/%Y %H:%M')
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
      self.df = self.df.set_index('datetime')
      # Create a snapshot for the raw dataset
      self.df_raw = self.df.copy()
      # Filter the dataset based on the start and end times
      self.df = self.df.loc[self.start_time:self.end_time]
    except Exception as e:
      raise Exception("Error pre-processing dataframe: " + dataset_name + ": " + str(e))
    
    # Narrow the dataset to the autodetected drying interval
    if autodetect_drying_interval:
      self.start_time, self.end_time = self.detect_drying_interval()
      self.df = self.df.loc[self.start_time:self.end_time]
    
  '''
  Function for loading in the 2024 weekly datasets
  Header structure:
  datetime, dryer_name, temp_high, temp_mid, temp_low, temp_out, temp
  '''
  def load_weekly_dataset_2024(self, dataset_name, start_time = None, end_time = None, autodetect_drying_interval = False):
    self.dataset_name = dataset_name
    self.start_time = start_time
    self.end_time = end_time
    self.grain_type = self.get_grain_type()

    print ("Loading dataset: " + str(self.dataset_folder)+"/"+dataset_name)
    print("Grain type identified as: " + self.grain_type)
    
    # Check if the dataset directory exists
    if not os.path.exists(self.dataset_folder):
      raise Exception("Datasets directory not found.")
    
    # Prepare dataset_local_path
    dataset_local_path = f'{self.dataset_folder}/{dataset_name}.csv'

    # Load the dataset into a pandas dataframe
    try:
      self.df = pd.read_csv(dataset_local_path)
    except Exception as e:
      raise Exception("Error loading dataset: " + str(e))
   
        # Column name remap for example datasets
    column_name_remap = {
      'temp1': 'temp_heated',
      'temp2': 'temp_high',
      'temp3': 'temp_mid',
      'temp4': 'temp_low',
      'temp5': 'temp_out',
    }

    # Drop the column with dryer name, rename the columns and set the datetime
    self.df = self.df.drop(self.df.columns[1], axis=1)
    self.df = self.df.rename(columns=column_name_remap)
 
    # Check for start and end times and initialize to entire dataset if not provided
    if self.start_time is None:
      self.start_time = self.df['datetime'][ self.df.index[0]]
    if self.end_time is None:
      self.end_time = self.df['datetime'][ self.df.index[-1]]

    # If timestamps are strings, convert to a pandas datetime objects
    if isinstance(self.start_time, str):
      self.start_time = pd.Timestamp(self.start_time)
    if isinstance(self.end_time, str):
      self.end_time = pd.Timestamp(self.end_time)

    print("Preprocessing dataset: %s, period: %s - %s" % (self.dataset_name, self.start_time.strftime("%H:%M"), self.end_time.strftime("%H:%M")))
    try:
      self.df['datetime'] = pd.to_datetime(self.df['datetime'])
      self.df = self.df.set_index('datetime')
      # Create a snapshot for the raw dataset
      self.df_raw = self.df.copy()
      # Filter the dataset based on the start and end times
      self.df = self.df.loc[self.start_time:self.end_time]

    except Exception as e:
      raise Exception("Error pre-processing dataframe: " + dataset_name + ": " + str(e))

  
  def plot_2024(self, raw=False, drier_temps_only=False):
    df = self.df_raw if raw else self.df

    # Visualize the dataframe
    if not drier_temps_only:
      # scatter, because data is scarce
      plt.plot(df.index, df["moist"], '-o' ,label="measured grain moisture", color="red")
      plt.plot(df.index, df["temp_heated"], label="temp_heated")
      plt.plot(df.index, df["temp_out"], label="temp_out")
    plt.plot(df.index, df["temp_mid"], label="temp_mid")
    plt.plot(df.index, df["temp_low"], label="temp_low")
    plt.plot(df.index, df["temp_high"], label="temp_high")
       
    # Draw vertical cursors for the beginning end end of the df if raw data is plotted
    if raw:
      plt.axvline(x=self.start_time, color='gray', linestyle='--', linewidth=1, label='start time')
      plt.axvline(x=self.end_time, color='gray', linestyle='--', linewidth=1, label='end time')

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    #make grid
    plt.grid(True)

    #set x axis  15 min
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))

    plt.ylabel("degrees in C and RH%")
    plt.title(self.dataset_name)
    plt.legend(loc='upper center', bbox_to_anchor=(1.02, 1.03))
    plt.show()


  # Autodetect start and end time for the drying cycle.
  # Algorithm:
  # Consider start time as the first derivative peak to the left of the first heated air temperature above 40C
  # Consider end time as the first derivative peak to the right of the last heated air temperature above 40C
  def detect_drying_interval(self):
    # Filter the dataset based on the start and end times
    df = self.df.copy()
    df = df.loc[self.start_time:self.end_time]
    
    # Filter the dataset to only include the heated air temperature
    df = df[df['temp_heated'] > 40]

    # Calculate the first derivative of the heated air temperature
    df['temp_heated_diff'] = df['temp_heated'].diff()

    # Find the first peak to the left of the first heated air temperature above 40C
    start_time = df[df['temp_heated_diff'] > 0].index[0]

    # Find the first peak to the right of the last heated air temperature above 40C
    end_time = df[df['temp_heated_diff'] < 0].index[-1]
  
    return start_time, end_time
  
  def extract_drying_cycles(self):
    # Filter the dataset to only include the heated air temperature
    df = self.df.copy()
    df = df.loc[self.start_time:self.end_time]

    # Add a column with the first derivative of the heated air temperature and apply smoothing
    df['temp_diff'] = df['temp_heated'].diff().rolling(window=3).mean()

    rise_threshold = 45  # The threshold temperature for the start of a cycle
    fall_threshold = 40  # The threshold temperature for the end of a cycle

    # Create a list to store the start and end times of cycles
    cycles = []
    
    # Initialize variables
    in_cycle = False
    start_time = None
    
    # Loop through the DataFrame rows
    for i in range(1, len(df)):
        temperature = df['temp_heated'].iloc[i]
        temp_diff = df['temp_diff'].iloc[i]
        timestamp = df.index[i]
        
        # Detect the point where the temperature rises above the rise_threshold
        if not in_cycle and temperature > rise_threshold:
            # Look back to find the point where the temperature started rising (temp_diff < 0)
            for j in range(i-1, 0, -1):
                if df['temp_diff'].iloc[j] < 0:
                    start_time = df.index[j+1]  # The point just after it started rising
                    if start_time is not pd.NaT:
                      in_cycle = True
                    else:
                      start_time = None
                    break
            # If no valid start_time is found, skip this cycle detection
            if start_time is None:
                continue
        
        # Detect the point where the temperature falls below the fall_threshold
        elif in_cycle and temperature < fall_threshold:
            # Look back to find the point where the temperature started falling (temp_diff > 0)
            end_time = None
            for j in range(i-1, 0, -1):
                if df['temp_diff'].iloc[j] > 0:
                    end_time = df.index[j+1]  # The point just after it started falling
                    if end_time is not pd.NaT:
                      cycles.append((start_time, end_time))
                      in_cycle = False
                      start_time = None
                    break
            
            # If no valid end_time is found, skip this cycle
            if end_time is None:
                continue
            
    return cycles
  
  def plot_weekly_data(self, cycles=[]):
    # Visualize the dataframe
    plt.plot(self.df.index, self.df["temp_heated"], label="temp_heated")
    plt.plot(self.df.index, self.df["temp_out"], label="temp_out")
    plt.plot(self.df.index, self.df["temp_mid"], label="temp_mid")
    plt.plot(self.df.index, self.df["temp_low"], label="temp_low")
    plt.plot(self.df.index, self.df["temp_high"], label="temp_high")

    # Draw vertical cursors for the beginning end of drying cycles if provided
    for cycle in cycles:
      plt.axvline(x=cycle[0], color='green', linestyle='--', linewidth=2)
      plt.axvline(x=cycle[1], color='red', linestyle='--', linewidth=2)
    
    # add legend labels for start and end cursors
    plt.plot([], [], color='green', linestyle='--', linewidth=2, label='cycle start')
    plt.plot([], [], color='red', linestyle='--', linewidth=2, label='cycle end')
       
    plt.gcf().autofmt_xdate()
    # Display only date
    myFmt = mdates.DateFormatter('%d-%m-%Y')
    plt.gca().xaxis.set_major_formatter(myFmt)

    #make grid
    plt.grid(True)

    #set x axis  1 day
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

    plt.ylabel("degrees in C and RH%")
    plt.title(self.dataset_name)
    plt.legend(loc='upper center', bbox_to_anchor=(1.02, 1.03))
    plt.show(block=False)
    return plt.gca()

  
  def get_ambient_temperature_array(self, t_eval):
      """
      Get the ambient temperature at the given time points. If the ambient temperature is not available at the given time points,
      the function will return the first or last available ambient temperature. In the future implementation, the function will
      include weather data or other sources to predict the ambient temperature.
      """
      # Get the beginning timestamp of the dataset
      start_time = self.df.index[0]

      # Interpolate the ambient temperature for t_eval using the dataset and convert from Celsius to Kelvin
      T_ambient_arr = np.interp(t_eval, (self.df.index - start_time).total_seconds(), self.df['temp_in'].values) + 273.15
      
      return T_ambient_arr