import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from model_parameters import ModelParameters


def heater_model(t, T_initial, T_final, time_constant):
    """
    Simulates the temperature change in a heater over time using a first-order exponential model. 

    Parameters: 
    t (float): The time at which to evaluate the temperature.
    T_initial (float): The initial temperature at time t=0.
    T_final (float): The final temperature as time approaches infinity.
    time_constant (float): The time constant of the system, which dictates the rate of temperature change.

    Returns:
    float: The temperature at time t.
    """

    T_air = T_initial + (T_final - T_initial) * (1 - np.exp(-t / time_constant))
    return T_air


def equilibrium_moisture_content(temp_K, relative_humidity, params):
    """
    Calculate the equilibrium moisture content (EMC) for a given temperature and 
    equilibrium relative humidity (ERH) using the Chung-Pfost model.
    Parameters:
    temp_K (float): Temperature in Kelvin.
    relative_humidity (float): Relative humidity as a percentage (0-100).
    params (dict): Dictionary containing the Chung-Pfost model parameters:
      - "A" (float): Parameter A of the Chung-Pfost model.
      - "B" (float): Parameter B of the Chung-Pfost model.
      - "C" (float): Parameter C of the Chung-Pfost model.
    Returns:
    float: Equilibrium moisture content as a decimal on a dry basis.
    """

    relative_humidity_decimal = relative_humidity / 100
    temp_C = temp_K - 273.15
    A = params["A"]
    B = params["B"]
    C = params["C"]
    
    # Check if the log argument is negative
    if (temp_C + C) <= 0:
        return 0

    emc = -1/B * np.log(-(temp_C + C)/A * np.log(relative_humidity_decimal))
    emc = max(emc, 0) # ensure emc is not negative
    emc /= 100  # Convert from percentage to fraction

    return emc # Moisture content decimal on dry basis

def drying_rate(grain_temp_K, params):
    """
    Calculate the drying rate of grain based on the Arrhenius equation.

    Parameters:
    grain_temp_K (float): The temperature of the grain in Kelvin.
    params (dict): A dictionary containing the following keys:
      - 'grain_pre_exponential_factor' (float): Pre-exponential factor (1/s).
      - 'grain_activation_energy' (float): Activation energy (J/mol).

    Returns:
    float: The drying rate (1/s).
    """

    k0 = params['grain_pre_exponential_factor']  # Pre-exponential factor (1/s)
    Ea = params['grain_activation_energy']  # Activation energy (J/mol)
    R = 8.314  # Ideal gas constant (J/(mol·K))

    # Calculate the drying rate based on Arrhenius equation
    k = k0 * np.exp(-Ea / (R * grain_temp_K))
    return k

 
def drying_ode_system(t, y, t_eval_sim, T_ambient_arr, params):
    """
    Defines the system of ordinary differential equations (ODEs) for the grain drying process.
    Parameters:
    t : float
      Current time step.
    y : list or array-like
      State variables at the current time step:
      [T_air_in, T_grain, M_grain, heat_in, heat_latent, heat_cooling]
      - T_air_in: Heated air temperature that enters the grain (K)
      - T_grain: Grain temperature (K)
      - M_grain: Moisture content of the grain (kg water / kg dry grain)
      - heat_in: Total energy transfered to the grain (J)
      - heat_latent: Total energy loss due to moisture evaporation (J)
      - heat_cooling: Total energy loss due to cooling (J)
    t_eval_sim : array-like
      Array of time points for which the system will be evaluated.
    T_ambient_arr : array-like
      Array of ambient temperatures corresponding to the time points in t_eval_sim.
    params : dict
      Dictionary of parameters required for the ODE system:
      - T_heater_setpoint: Heater setpoint temperature (K)
      - heater_tau: Time constant for the heater (s)
      - grain_volume_drier: Volume of grain in the drier (m³)
      - grain_volume_total: Total volume of grain (m³)
      - cp_grain: Specific heat capacity of the grain (J/(kg·K))
      - ua_conv_air_to_grain: Heat transfer coefficient from air to grain (W/K)
      - ua_conv_cooling: Heat transfer coefficient for cooling (W/K)
      - grain_emc_params: Parameters for the equilibrium moisture content model
      - rho_grain: Density of the grain (kg/m³)
      - latent_heat_vaporization: Latent heat of vaporization of water (J/kg)
    Returns:
    list
      Derivatives of the state variables:
      [dT_air_in_dt, dT_grain_dt, dM_dt, heat_rate_in, heat_rate_latent, heat_rate_cooling]
      - dT_air_in_dt: Rate of change of air inlet temperature (K/s)
      - dT_grain_dt: Rate of change of grain temperature (K/s)
      - dM_dt: Rate of change of moisture content (kg water / kg dry grain / s)
      - heat_rate_in: Heat input rate (W)
      - heat_rate_latent: Latent heat loss rate (W)
      - heat_rate_cooling: Cooling heat loss rate (W)
    """
    
    # Unpack the state variables
    T_air_in, T_grain, M_grain, heat_in, heat_latent, heat_cooling = y

    # Interpolate the ambient temperature for the current time step
    t_idx = np.searchsorted(t_eval_sim, t)
    # Check that the index is within the bounds of the ambient temperature array
    t_idx = min(max(t_idx, 0), len(T_ambient_arr) - 1)
    T_ambient = T_ambient_arr[t_idx]

    # Unpack parameters
    T_heater_setpoint = params['T_heater_setpoint']
    heater_tau = params['heater_tau']
    #print("Heater setpoint: ", T_heater_setpoint, "Heater tau: ", heater_tau)
    grain_volume_drier = params['grain_volume_drier']
    grain_volume_total = params['grain_volume_total']
    C_p = params['cp_grain']
    ua_conv_air_to_grain = params['ua_conv_air_to_grain']
    ua_conv_cooling = params['ua_conv_cooling']
    grain_emc_params = params['grain_emc_params']
    rho_grain = params['rho_grain']  # Grain density (kg/m³)
    latent_heat_vaporization = params['latent_heat_vaporization']  # Latent heat of vaporization of water (J/kg)
    
    grain_mass_total = rho_grain * grain_volume_total  # Mass of grain in kg
    grain_mass_drier = rho_grain * grain_volume_drier  # Mass of grain in the drier (kg)
    grain_mass_buffer = max(grain_mass_total - grain_mass_drier, 0)  # Mass of grain in the buffer (kg), ensure non-negative

    # Predict the air inlet temperature or use measurement history
    #T_air_in = heater_model(t, T_ambient, T_heater_setpoint, heater_tau) # now part of the state variables
    dT_air_in_dt = (T_heater_setpoint - T_air_in) / heater_tau

    # Compute equilibrium moisture content for the given grain temperature and relative humidity (%)
    M_eq = equilibrium_moisture_content(T_grain, params['rh_air_in'], grain_emc_params)

    # Compute drying rate for given grain temperature
    k = drying_rate(T_grain, params)

    # Compute the local drying rate (change in moisture content) using thin-layer drying model
    dM_dt = -k * (M_grain - M_eq) # kg water / kg dry grain / s

    # Calculate water mass evaporation rate
    dM_water_mass_dt = dM_dt * grain_mass_drier  # kg/s

    # Compute latent heat loss due to moisture evaporation
    heat_rate_latent = latent_heat_vaporization * dM_water_mass_dt  # Latent heat loss in W (J/s)

    # Compute heat transfer rate due to cooling of the grain to the ambient temperature
    heat_rate_cooling = ua_conv_cooling * (T_grain - T_ambient)

    # Calculate heat transfer rate in the heated region
    heat_rate_in = ua_conv_air_to_grain * (T_air_in - T_grain)

    # Update grain temperature change rate
    dT_grain_dt = heat_rate_in / (grain_mass_drier * C_p)         # rate change due to air heating the grain
    dT_grain_dt += heat_rate_latent / (grain_mass_drier * C_p)    # rate change due to latent heat loss
    dT_grain_dt += heat_rate_cooling / (grain_mass_buffer * C_p)  # rate change due to ambient cooling

    # Return the derivative of the state variables
    return [dT_air_in_dt, dT_grain_dt, dM_dt, heat_rate_in, heat_rate_latent, heat_rate_cooling]


def simulate_model_base(t_eval_sim, T_ambient_arr, params: ModelParameters):
    # Extract parameters from the dictionary
    M_initial = params['M_initial']  # Initial moisture content of the grain (kg water / kg dry grain)
    T_air_in_initial = T_ambient_arr[0]  # Initial air inlet temperature (K)
    T_grain_initial = T_ambient_arr[0]  # Initial grain temperature (K)
    
    # Set initial conditions for the ODE system
    y0 = [T_air_in_initial, T_grain_initial, M_initial, 0, 0, 0]

    # Time span for the simulation
    t_span = (t_eval_sim[0], t_eval_sim[-1])
    
    # Solve the ODE system
    sol = solve_ivp(lambda t, y: drying_ode_system(t, y, t_eval_sim, T_ambient_arr, params), t_span, y0, t_eval=t_eval_sim, method='RK45')

    # Calculate the air inlet temperature
    #T_air_in = np.array([heater_model(t, T_air_ambient, params['T_heater_setpoint'], params['heater_tau']) for t in t_eval_sim])
    
    # Extract the solution
    T_air_in = sol.y[0]  # Air inlet temperature (K)
    T_grain = sol.y[1] # Temperature of the grain (K)
    M_grain = sol.y[2] # Moisture content of the grain (kg water / kg dry grain)

    # Extract the heat transfer rates
    heat_in = sol.y[3]      # Heat input to the grain (J)
    heat_latent = sol.y[4]  # Latent heat loss due to moisture evaporation (J)
    heat_cooling = sol.y[5] # Heat loss due to cooling of the grain (J)


    # pack simulation results into a dictionary
    results = {
        't_eval': t_eval_sim,
        'T_air_in': T_air_in,
        'T_grain': T_grain,
        'M_grain': M_grain,
        'heat_in': heat_in,
        'heat_latent': heat_latent,
        'heat_cooling': heat_cooling
    }
    return results

def event_grain_moisture_target_reached(t, y, target_moisture):
    # Unpack the state variables
    T_air_in, T_grain, M_grain, heat_in, heat_latent, heat_cooling = y

    # Check if the target moisture content has been reached
    return M_grain - target_moisture

event_grain_moisture_target_reached.terminal = True  # Stop the simulation when the target moisture content is reached
event_grain_moisture_target_reached.direction = -1  # Only trigger when the target moisture content is reached from above

def simulate_model_base_with_events(ax, t_eval_sim, T_ambient_arr, params: ModelParameters, target_moisture):
    M_initial = params['M_initial']  # Initial moisture content of the grain (kg water / kg dry grain)
    T_air_in_initial = T_ambient_arr[0]  # Initial air inlet temperature (K)
    T_grain_initial = T_ambient_arr[0]  # Initial grain temperature (K)
    
    # Set initial conditions for the ODE system
    y0 = [T_air_in_initial, T_grain_initial, M_initial, 0, 0, 0]

    # Time span for the simulation
    t_span = (t_eval_sim[0], t_eval_sim[-1])
    
    # Solve the ODE system
    sol = solve_ivp(lambda t, y: drying_ode_system(t, y, t_eval_sim, T_ambient_arr, params),
                    t_span,
                    y0,
                    t_eval=t_eval_sim,
                    method='RK45',
                    events=lambda t, y: event_grain_moisture_target_reached(t, y, target_moisture))
    
    # None indicates that the target moisture content was not reached
    t_target = None
    M_target = None
    
    # Check if the target moisture content was reached
    if sol.t_events[0].size > 0:
        # Target moisture content was reached
        t_target = sol.t_events[0][0]
        M_target = sol.y_events[0][0][2]

    return t_target, M_target

def plot_temperature_evolution(T, params):
  # Convert the temperature data to Celsius
  T = T - 273.15
  
  total_volume = params['grain_volume_total']  # Total volume of the grain (m³)
  channel_area = params['grain_surface_area']  # Cross-sectional area of the channel (m²)
  
  L = total_volume / channel_area  # Length of the channel (meters) i.e approx facility height
  Nt = T.shape[0]  # Total number of time steps

  # Define positions for the line plot
  positions_to_plot = np.arange(0, L, 1)  # Every meter along the conveyor
  indices_to_plot = [int(pos / params['dx']) for pos in positions_to_plot]

  # Create a figure with two subplots
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

  # Heatmap plot
  cax1 = ax1.imshow(np.flip(T.T, axis=0), aspect='auto', cmap='plasma', origin='lower', extent=[0, Nt*params['dt'], L, 0])
  fig.colorbar(cax1, ax=ax1, label='Temperature (°C)')
  ax1.set_xlabel('Time (s)')
  ax1.set_ylabel('Position (m)')
  ax1.set_title('Temperature of a transported grain over time')
  ax1.axhspan(params['drier_x_start'], params['drier_x_end'], color='red', alpha=0.3, label='Heated Region')
  ax1.legend()

  # Line plot at specific positions
  for idx, pos in zip(indices_to_plot, positions_to_plot):
    if pos == 0:
      ax2.plot(np.arange(0, Nt*params['dt'], params['dt']), T[:, idx], label=f'Position {pos:.1f} and {L:.1f} m (boundaries)', color='gray')
    else:
      ax2.plot(np.arange(0, Nt*params['dt'], params['dt']), T[:, idx], label=f'Position {pos:.1f} m')

  ax2.set_xlabel('Time (s)')
  ax2.set_ylabel('Temperature (°C)')
  ax2.set_title('Temperature Evolution at Different Positions')
  ax2.legend()
  ax2.grid(True)

  

  # Show the combined figure
  plt.tight_layout()
  plt.show(block=False)

  # move figure to the rightmost screen
  fig.canvas.manager.window.geometry('+4000+0')
  plt.show()

def plot_temperature_over_time(T, params):
    """
    Plot temperature over time for specific positions along the channel.

    Parameters:
    - T: numpy array of shape (Nt, Nx) containing temperature data
    - params: dictionary containing the simulation parameters
    """
    Nx = params['Nx']  # Number of spatial grid points
    time_steps = T.shape[0]
    drier_x_start = params['drier_x_start']
    drier_x_end = params['drier_x_end']
    heated_start_idx = int(drier_x_start / (params['total_volume'] / params['A_channel']) * Nx)
    heated_end_idx = int(drier_x_end / (params['total_volume'] / params['A_channel']) * Nx)
    
    # Define time coordinate
    times = np.arange(time_steps) * params['dt']

    # Define specific positions at the start, middle, and end of the heated region
    positions_to_plot = [heated_start_idx, int((heated_start_idx + heated_end_idx) / 2), heated_end_idx]
    position_values = np.linspace(0, params['total_volume'] / params['A_channel'], Nx)[positions_to_plot]
    labels = [f'Heated Start ({position_values[0]:.2f} m)', f'Heated Mid ({position_values[1]:.2f} m)', f'Heated End ({position_values[2]:.2f} m)']
    
    plt.figure(figsize=(12, 6))
    
    for pos_idx in positions_to_plot:
        plt.plot(times, T[:, pos_idx], label=labels.pop(0))

    plt.title('Temperature Over Time at Different Positions Along the Channel')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()


"""
Plot the temperature at the top, center, and bottom end of heated region (temp_high, temp_mid, temp_low) and compare it with the experimental dataset.
"""
def plot_temp_comparison(T, params, exp_dataset):
  # Convert the temperature data to Celsius
  T = T - 273.15
  # Extract the temperature data from the simulation

  L = params['total_volume'] / params['channel_area']  # Length of the channel (meters)
  Nx = T.shape[1]  # Number of spatial grid points
  drier_x_start = params['drier_x_start']
  drier_x_end = params['drier_x_end']
  drier_x_mid = (drier_x_start + drier_x_end) / 2
  heated_start_idx = int(drier_x_start / L * Nx)
  heated_end_idx = int(drier_x_end / L * Nx)
  heated_mid_idx = int(drier_x_mid / L * Nx)

  temp_high = T[:, heated_start_idx]  # Temperature at the top end of the heated region
  temp_mid = T[:, heated_mid_idx]  # Temperature at the center of the heated region
  temp_low = T[:, heated_end_idx]  # Temperature at the bottom end of the heated region

  # Extract the experimental data
  df = exp_dataset.df
  time = df.index
  temp_high_exp = df['temp_high'].values
  temp_mid_exp = df['temp_mid'].values
  temp_low_exp = df['temp_low'].values

  # Generate pandas time array for the simulation that starts from the beginning of the experimental data
  time_sim = time[0] + pd.to_timedelta(np.arange(len(temp_high)) * params['dt'], unit='s')

  # clear the current figure with the given reference name
  plt.close('temp_comparison')
  # Create a figure with given reference name
  fig = plt.figure('temp_comparison', figsize=(12, 6))

  # Line plot of temperature at the top end of the heated region with coordinate labels
  plt.plot(time_sim, temp_high, label='Top (sim) x=%.2f m' % drier_x_start, color='blue')
  plt.plot(time, temp_high_exp, label='Top (exp) x=%.2f m' % drier_x_start, linestyle='dashed', color='blue')
  plt.plot(time_sim, temp_mid, label='Mid (sim) x=%.2f m' % drier_x_mid, color='red')
  plt.plot(time, temp_mid_exp, label='Mid (exp) x=%.2f m' % drier_x_mid, linestyle='dashed', color='red')
  plt.plot(time_sim, temp_low, label='Bottom (sim) x=%.2f m' % drier_x_end, color='green')
  plt.plot(time, temp_low_exp, label='Bottom (exp) x=%.2f m' % drier_x_end, linestyle='dashed', color='green')
  plt.xlabel('Time')
  plt.ylabel('Temperature (°C)')
  plt.title('Temperature at the Top, Mid, and Bottom End of the Heated Region')
  plt.legend()

  # Format time axis
  plt.gcf().autofmt_xdate()
  myFmt = mdates.DateFormatter('%H:%M')
  plt.gca().xaxis.set_major_formatter(myFmt)
  
  # Show the combined figure
  plt.tight_layout()
  plt.show(block=False)

  # move figure to the rightmost screen
  fig.canvas.manager.window.geometry('+4000+0')
  plt.show()