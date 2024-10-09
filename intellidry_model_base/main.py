"""
main.py

This script is part of the IntelliDry project, which aims to develop an AI-based process control system for grain dryer automation.

Usage:
  This script can be run directly to perform interactive model testing, optimization, or visualization of the fit.

Author:
  Veiko Vunder <veiko.vunder@ut.ee>

Year:
  2014
"""

from dataset import Dataset
from model_parameters import ModelParameters
from model_base import simulate_model_base, simulate_model_base_with_events
from scipy.optimize import differential_evolution
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import ttk

from grain_params import grain_emc_params


def create_figure_fit_comparison():
  """
  Create the initial figure for temperature comparison between simulation and experimental data.
  """
  # Create a new figure with a given reference name and size
  fig, ax = plt.subplots(figsize=(12, 6), num='temp_comparison')

  # Set axis labels and title
  ax.set_xlabel('Time')
  ax.set_ylabel('Temperature (°C)')
  ax.set_title('Temperature at the Top, Mid, and Bottom End of the Heated Region')

  # Format time axis
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
  plt.gcf().autofmt_xdate()
  
  # Add legend
  ax.legend()

  # Show the figure
  plt.tight_layout()
  plt.show(block=False)

  # Move figure to the rightmost screen
  fig.canvas.manager.window.geometry('+4000+0')
  
  return ax


def create_figure_convergence():
  fig, ax_conv = plt.subplots(figsize=(12, 6), num='convergence')
  ax_conv.set_xlabel('Iteration')
  ax_conv.set_ylabel('Mean Squared Error')
  ax_conv.set_title('Convergence of the Differential Evolution Optimization')
  ax_conv.set_yscale('log')
  ax_conv.grid(True)
  plt.tight_layout()
  plt.show(block=False)
  return ax_conv


def update_figure_fit_comparison(ax, params, sim_results, exp_dataset):
  """
  Update the figure for the fit comparison between simulation and experimental data.
  """
  # unpack simulation results
  T_air_in = sim_results['T_air_in']
  M_grain = sim_results['M_grain']
  T_grain_sim_K = sim_results['T_grain']

  # Extract the experimental data
  df = exp_dataset.df
  time_exp = df.index
  #temp_high_exp = df['temp_high'].values
  #temp_mid_exp = df['temp_mid'].values
  #temp_low_exp = df['temp_low'].values
  T_grain_exp_C = df['temp_low'].values

  # Generate pandas time array for the simulation starting from the beginning of the experimental data
  time_sim = time_exp[0] + pd.to_timedelta(sim_results['t_eval'], unit='s')
  
  # Clear the axes to update the plot
  ax.clear()

  # Line plot of temperature at different positions with coordinate labels
  ax.plot(time_sim, T_grain_sim_K - 273.15, label='T_grain_sim', color='red')
  ax.plot(time_exp, T_grain_exp_C, label='T_grain_exp', linestyle='dotted', color='red')
  ax.plot(time_exp, df['temp_heated'], label='Air inlet exp', linestyle='dotted', color='black')
  ax.plot(time_sim, (T_air_in - 273.15), label='Air Inlet sim', linestyle='dashdot', color='black')
  ax.plot(time_exp, df['moist'], '-o', label='Moisture %', color='orange')
  ax.plot(time_sim, M_grain*100, label='Moisture sim %', linestyle='dashdot', color='green')
  
  # evap water mass 
  #ax.plot(time_sim, sim_results['evap_water_mass'], label='Evap water mass sim (kg)', linestyle='dashdot', color='blue')
  #ax.plot(time_sim, sim_results['heat_in'], label='heat_in', linestyle='dashdot', color='red')
  #ax.plot(time_sim, sim_results['heat_cooling'], label='heat_cooling', linestyle='dashdot', color='lightblue')
  #ax.plot(time_sim, sim_results['heat_latent'], label='heat_latent', linestyle='dashdot', color='blue')

  # Update axis labels and title
  ax.set_xlabel('Time')
  ax.set_ylabel('Temperature (°C)')
  ax.set_title('Base model simulation vs. experimental data')

  # Update legend
  ax.legend()

  # Format time axis
  plt.gcf().autofmt_xdate()
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
  ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))

  # Redraw the updated plot
  ax.figure.canvas.draw()

  # Add grid
  ax.grid(True)
    

def update_plot(*args, ax, exp_dataset):
  """
  Update the plot based on the slider values.
  """

  global sliders

  # Read current slider values and update params
  params = {
    'T_heater_setpoint': sliders['T_heater_setpoint'].get(),
    'heater_tau': sliders['heater_tau'].get(),
    'grain_volume_total': sliders['grain_volume_total'].get(),
    'grain_volume_drier': sliders['grain_volume_drier'].get(),
    'dt': sliders['dt'].get(),
    't_end': sliders['t_end'].get(),
    'T_ambient': sliders['T_ambient'].get(),
    'rh_air_in': sliders['rh_air_in'].get(),
    'ua_conv_air_to_grain': sliders['ua_conv_air_to_grain'].get(),
    'cp_grain': sliders['cp_grain'].get(),
    'rho_grain': sliders['rho_grain'].get(),
    'ua_conv_cooling': sliders['ua_conv_cooling'].get(),
    'M_initial': sliders['M_initial'].get(),
    'grain_pre_exponential_factor': sliders['grain_pre_exponential_factor'].get(),
    'grain_activation_energy': sliders['grain_activation_energy'].get(),
    'grain_emc_params': grain_emc_params['wheat'], #TODO: hardcoded for wheat
    'latent_heat_vaporization': 2.257e6  # J/kg
  }
    
  #Create a time array for simulation using exactly the same timestamps as the experimental data in seconds from the beginning of measurements
  t_eval_sim = np.arange(0, sliders['t_end'].get(), sliders['dt'].get())
  T_ambient_arr = np.ones_like(t_eval_sim) * params['T_ambient']

  # Simulate the model with the given parameters
  sim_results = simulate_model_base(t_eval_sim, T_ambient_arr, params)
  
  # Plot updated temperature evolution
  #plot_temperature_over_time2(ax, T, params)
  #plot_temp_comparison(T, params, ds_peas1)
  update_figure_fit_comparison(ax, params, sim_results, exp_dataset)


# Helper function to create a slider
def create_slider(root, label, from_, to, resolution, length, default_value):
  '''
  Create a slider with the given parameters and pack it into the root window.
  '''

  slider = tk.Scale(root, from_=from_, to=to, resolution=resolution, length=length, label=label, orient=tk.HORIZONTAL)
  slider.set(default_value)
  slider.pack()
  return slider

def run_interactive_model_testing(exp_dataset):
  '''
  Run the interactive model testing GUI.
  '''
  global sliders

  # Initialize Tkinter window
  root = tk.Tk()
  root.title("Simulation Parameters")

  # Define slider length
  slider_length = 800  # Adjust this value to make the sliders wider
  
  # Create sliders for each parameter with +-50% range of default values
  sliders['T_heater_setpoint'] = create_slider(root, 'Heater Setpoint Temperature (K)', 273.15 + 60, 273.15 + 90, 1.0, slider_length, 273.15 + 70)
  sliders['T_ambient'] = create_slider(root, 'Ambient Temperature (K)', 273.15 + 5, 273.15 + 30, 1.0, slider_length, 273.15 + 25)
  sliders['rh_air_in'] = create_slider(root, 'Relative Humidity of Air in Drier', 0.1, 0.9, 0.01, slider_length, 0.4)
  sliders['heater_tau'] = create_slider(root, 'Heater Time Constant (s)', 60, 600, 10, slider_length, 100)
  sliders['grain_volume_total'] = create_slider(root, 'Grain Volume Total (m³)', 15.0, 40.0, 1.0, slider_length, 28.0)
  sliders['grain_volume_drier'] = create_slider(root, 'Grain Volume in Drier (m³)', 5.0, 15.0, 1.0, slider_length, 14.0)
  sliders['dt'] = create_slider(root, 'Time Step Size (s)', 0.01, 1000, 0.1, slider_length, 5.0)
  sliders['t_end'] = create_slider(root, 'Total Simulation Time (s)', 1*3600, 10*3600, 300, slider_length, 8*3600)
  sliders['ua_conv_air_to_grain'] = create_slider(root, 'Overall Heat Transfer Coefficient (air->grain) (W/K)', 1.0e4, 2.5e4, 1e2, slider_length, 1.0e4)
  sliders['cp_grain'] = create_slider(root, 'Specific Heat Capacity of Grain (J/kgK)', 1000, 4000, 50, slider_length, 2550)
  sliders['rho_grain'] = create_slider(root, 'Density of Grain (kg/m³)', 400, 1200, 50, slider_length, 800)
  sliders['ua_conv_cooling'] = create_slider(root, 'Overall Cooling Coefficient (W/K)', -2e4, -1, 1, slider_length, -2e3)
  sliders['M_initial'] = create_slider(root, 'Initial Moisture Content (kg/kg)', 0.1, 0.3, 0.01, slider_length, 0.2)
  sliders['grain_pre_exponential_factor'] = create_slider(root, 'Pre-exponential Factor', 0.1, 100, 0.01, slider_length, 6.0)
  sliders['grain_activation_energy'] = create_slider(root, 'Activation Energy', 1e4, 1e5, 1e3, slider_length, 3.0e4)

  ax = create_figure_fit_comparison()
  update_plot_with_args = partial(update_plot, ax=ax, exp_dataset=exp_dataset) # pass additional arguments to the update callback
  
  # Attach update function to sliders
  for key in sliders:
    sliders[key].config(command=lambda args, key=key: update_plot_with_args())
  # update plot
  update_plot(ax=ax, exp_dataset=exp_dataset)
  # Start the GUI loop
  root.mainloop()


def de_objective(param_array, exp_dataset):
  '''
  Objective function for differential evolution optimization.
  '''
  
  # Extract the parameters from the parameter array into our params object
  params.update_optimized_parameters(param_array)
  #print optimized parameters
  #print('\n'.join([f'{key}: {value}' for key, value in params.optimized_parameters.items()]))
  # print array
  #print(param_array)
  #print ("--------------------------------------------")
  
  

  #Create a time array for simulation using exactly the same timestamps as the experimental data in seconds from the beginning of measurements
  t_eval_sim = (exp_dataset.df.index - exp_dataset.df.index[0]).total_seconds().values
  T_ambient_arr = exp_dataset.df['temp_in'].values + 273.15
  T_air_exp_heated = exp_dataset.df['temp_heated'].values + 273.15

  # Simulate the model with the given parameters
  sim_results = simulate_model_base(t_eval_sim, T_ambient_arr, params)
  
  # Extract the simulated and experimental grain temperatures
  T_air_in = sim_results['T_air_in'] # include this in mse to optimize heater model params
  T_grain_sim = sim_results['T_grain']
  T_grain_exp = exp_dataset.df['temp_low'].values + 273.15 # load values and convert to K
  M_grain_percent = sim_results['M_grain'] * 100
  M_grain_exp_percent = exp_dataset.df['moist'].values
  
  # Calculate the mean squared error between the simulated and experimental grain temperature, ignore nan values
  mse_T = np.nanmean((T_grain_sim - T_grain_exp)**2)
  mse_T0 = np.nanmean((T_grain_sim[0] - T_grain_exp[0])**2)
  mse_M = np.nanmean((M_grain_percent - M_grain_exp_percent)**2)
  mse_T_air_in = np.nanmean((T_air_in - T_air_exp_heated)**2)
  mse = mse_T + mse_M * 100 + mse_T0 * 100 + mse_T_air_in # penalize the first time step and moisture content more
  
  #return mse_T_air_in + mse_T
  return mse



def de_callback(xk, convergence, convergences, ax_fit, exp_dataset):
  '''
  Callback function for differential evolution optimization. Called after each iteration.
  '''
  viz_params = params
  viz_params.update_optimized_parameters(xk)

  #Create a time array for simulation using exactly the same timestamps as the experimental data in seconds from the beginning of measurements
  t_eval_sim = (exp_dataset.df.index - exp_dataset.df.index[0]).total_seconds().values
  T_ambient_arr = exp_dataset.get_ambient_temperature_array(t_eval_sim)

  # Simulate the model with the given parameters
  sim_results = simulate_model_base(t_eval_sim, T_ambient_arr, viz_params)
  
  # Extract the simulated and experimental grain temperatures
  T_grain_sim_K = sim_results['T_grain']
  T_grain_exp_K = exp_dataset.df['temp_low'].values + 273.15 # load values and convert to K
  M_grain_percent = sim_results['M_grain'] * 100
  M_grain_exp_percent = exp_dataset.df['moist'].values
  
  # Calculate the mean squared error between the simulated and experimental grain temperature, ignore nan values
  mse_T = np.nanmean((T_grain_sim_K - T_grain_exp_K)**2)
  mse_M = np.nanmean((M_grain_percent - M_grain_exp_percent)**2)
  mse = mse_T + mse_M * 100
  
  # Append the convergence data to the dictionary
  convergences["current"].append(mse)

  # Debugging
  #print(f'MSE_T: {mse_T}, MSE_M: {mse_M}, MSE: {mse}')
  #print('\n'.join([f'{key}: {value}' for key, value in params.optimized_parameters.items()]))
  #print("--------------------------------------------")

  # show the fit comparison plot and pause for a while to let the GUI update
  update_figure_fit_comparison(ax_fit, viz_params, sim_results, exp_dataset)
  plt.pause(0.2)


def compare_de_strategies(exp_dataset, params, ax_fit, ax_conv):
  # Convert opt_param_bounds to a list of (min, max) tuples for differential_evolution
  bounds = [(min_val, max_val) for min_val, max_val in params.optimized_parameters_bounds.values()]

  # Datastructure to store the convergence for different strategies of each iteration
  convergences = {}

  # Run differential evolution optimization with all available strategies and plot the results
  for strategy in ['best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin', 'rand2bin']:
    print(f"Running optimization with strategy: {strategy}")
    convergences["current"] = []
    de_callback_with_args = partial(de_callback, convergences, ax_fit, exp_dataset) # pass additional arguments to the callback function
    res = differential_evolution(de_objective, bounds, maxiter=40, disp=True, workers=1, strategy=strategy, callback=de_callback_with_args, args=(exp_dataset,))
    print(f"Optimized parameters with strategy {strategy}:")
    params.update_optimized_parameters(res.x)
    print('\n'.join([f'{key}: {value}' for key, value in params.optimized_parameters.items()]))
    convergences[strategy] = convergences["current"]
    convergences["current"] = []
    # Update convergence plot
    ax_conv.plot(convergences[strategy], label=strategy)
    ax_conv.legend()
    plt.pause(0.2)

def run_optimization(exp_dataset, params):
  # Convert opt_param_bounds to a list of (min, max) tuples for differential_evolution
  bounds = [(min_val, max_val) for min_val, max_val in params.optimized_parameters_bounds.values()]

  # Initialize the figure for fit
  ax_fit = create_figure_fit_comparison()

  # Create figure for convergence plot
  ax_conv = create_figure_convergence()

  #print(f"Running optimization with strategy: {strategy}")
  # Datastructure to store the convergence for different strategies of each iteration
  convergences = {}
  convergences["current"] = []
  strategy = 'best1bin'
  de_callback_with_args = partial(de_callback, convergences=convergences, ax_fit=ax_fit, exp_dataset=exp_dataset) # pass additional arguments to the callback function
  res = differential_evolution(de_objective, bounds, maxiter=150, disp=True, workers=-1, strategy=strategy, callback=de_callback_with_args, args=(exp_dataset,))
  print(f"Optimized parameters with strategy {strategy}:")
  params.update_optimized_parameters(res.x)
  print('\n'.join([f'{key}: {value}' for key, value in params.optimized_parameters.items()]))
  convergences[strategy] = convergences["current"]
  convergences["current"] = []
  # Update convergence plot
  ax_conv.plot(convergences[strategy], label=strategy)
  ax_conv.legend()
  plt.pause(0.2)

  # Write best params, opt params and opt_param_bounds to a json file at the same location as the dataset
  params.save()

  # Save the figure containing ax axis to a file
  ax_fit.get_figure().savefig(f'{exp_dataset.dataset_folder}/{exp_dataset.dataset_name}_fit.png')


  # Perform optimization using differential evolution
  #result = differential_evolution(objective, bounds, maxiter=100, disp=True, workers=4, strategy='randtobest1exp' , callback=diff_evolution_callback)

  # Extract the optimized parameters
  #params.update_optimized_parameters(result.x)

  # Print the optimized parameters
  #print("Optimized parameters:")
  #print('\n'.join([f'{key}: {value}' for key, value in params.optimized_parameters.items()]))

  
'''
Loads file with best fit parameters and calls update_figure_temp_comparison()
to compare simulation results with experimental data.
'''
def fit_visualizer(ds):
  # Load the parameters from the json file
  params = ModelParameters()
  params.load(ds.dataset_name)

  #Create a time array for simulation using exactly the same timestamps as the experimental data in seconds from the beginning of measurements
  t_eval_sim = (ds.df.index - ds.df.index[0]).total_seconds().values
  T_ambient_arr = ds.get_ambient_temperature_array(t_eval_sim)

  # Initialize the figure
  ax_fit = create_figure_fit_comparison()
  # Update the figure with the optimized parameters
  update_figure_fit_comparison(ax_fit, params, simulate_model_base(t_eval_sim, T_ambient_arr, params), ds)
  
  # Start the GUI loop
  plt.show()

'''
Estimates the drying duration for the given dataset and fitted parameters.
'''
def estimate_drying_duration(ds, target_moisture, plot=False):
  # Load the parameters from the json file
  params = ModelParameters()
  params.load(ds.dataset_name)
  
  ax = None
  if plot:
    ax = create_figure_fit_comparison()

  t_max = 12 * 3600 # 12 hours
  t_eval_sim = np.arange(0, t_max, 60) # 1 minute time steps
  T_ambient_arr = ds.get_ambient_temperature_array(t_eval_sim)
  
  # Simulate the model with the given parameters
  t_target, M_target = simulate_model_base_with_events(ax, t_eval_sim, T_ambient_arr, params, target_moisture)
  if t_target is None:
    print(f'Target moisture content {target_moisture} not reached within {t_max/3600:.2f} hours, current moisture content {M_target*100:.2f}%')
  else:
    print(f'Target moisture content {target_moisture} reached at {t_target/3600:.2f} hours, current moisture content {M_target*100:.2f}%')
  
  if plot:
    # Start the GUI loop
    update_figure_fit_comparison(ax, params, simulate_model_base(t_eval_sim, T_ambient_arr, params), ds)
    plt.show()


####################################################################################
####################################   M A I N    ##################################
####################################################################################
## Choose the experimental dataset to load by uncommenting the following sections ##
####################################################################################

#exp_dataset = Dataset()
#exp_dataset.load_example_dataset('buckwheat_2023_10_19', start_time='2023-10-19 10:00:00', end_time='2023-10-19 21:00:00', autodetect_drying_interval=True)
#exp_dataset.plot(raw=False, drier_temps_only=True)

exp_dataset = Dataset()
exp_dataset.load_example_dataset('oats_2023_09_11', start_time='2023-09-11 14:52:00', end_time='2023-09-11 16:05:00')
#exp_dataset.plot(raw=False, drier_temps_only=True)

#exp_dataset = Dataset()
#exp_dataset.load_example_dataset('peas_2023_09_08', start_time='2023-09-08 11:03:45', end_time='2023-09-08 17:50:00')
#exp_dataset.plot(raw=False, drier_temps_only=True)


# Dicionary to store the slider objects
sliders = {}

# Define the model parameters
params = ModelParameters()

# Initialize the parameters with the default values
params.dataset_name = exp_dataset.dataset_name
params.set_const("grain_type", exp_dataset.grain_type)
#params.set_const("T_heater_setpoint", 353.15)
#params.set_const("heater_tau", 100)
params.set_const("rho_grain", 800)
params.set_const("latent_heat_vaporization", 2.257e6)
params.set_const("grain_emc_params", grain_emc_params[exp_dataset.grain_type])
params.set_const("rh_air_in", 0.4) # relative humidity of the air in the drier, TODO: not a constant and should be calculated

# Set the optimized parameters initial values with bounds (cannot contain sub-dictionaries)
params.set_opt("T_heater_setpoint", 273.15 + 70, (273.15 + 60, 273.15 + 90))
params.set_opt("heater_tau", 100, (60, 300))
params.set_opt("grain_volume_total", 28.0, (15.0, 40.0))
params.set_opt("grain_volume_drier", 14.0, (5.0, 15.0))
# params.set_opt("T_ambient", 273.15 + 20, (273.15 + 0, 273.15 + 30))
params.set_opt("cp_grain", 2550, (1000, 4000))
params.set_opt("ua_conv_air_to_grain", 1.0e4, (1.0e4, 2.5e4))
params.set_opt("ua_conv_cooling", -2e3, (-2e4, -1))
params.set_opt("M_initial", 0.2, (0.1, 0.3))
params.set_opt("grain_pre_exponential_factor", 6.0, (0.1, 100))
params.set_opt("grain_activation_energy", 3.0e4, (1e4, 1e5))

# Overwrite defaults by loading the parameters from the json file
# Enabling this will start the optimization from the last saved parameters
try:
  params.load(exp_dataset.dataset_name)
except:
  print("No saved parameters found, using default values.")

############################################################################
## Choose the action to perform by uncommenting the desired function call ##
############################################################################
run_interactive_model_testing(exp_dataset)
#run_optimization(exp_dataset, params)
#fit_visualizer(exp_dataset)
#estimate_drying_duration(exp_dataset, 0.12, plot=True)

# Stay in the GUI loop
plt.show()