# solar_models.py
import pymc as pm
import numpy as np

def build_dynamic_model(df, config):
    """
    Builds a PyMC model dynamically based on a loaded JSON configuration.
    """
    with pm.Model() as model:
        # --- 1. COMMON BASELINE ELEMENTS ---
        alpha = pm.Normal('alpha', mu=0, sigma=0.1)
        beta_weekend = pm.Normal('beta_weekend', mu=-1, sigma=0.3)
        
        # Seasonality
        sin_matrix = df[['sin_k1', 'sin_k2', 'sin_k3']].values
        cos_matrix = df[['cos_k1', 'cos_k2', 'cos_k3']].values
        gamma = pm.Normal('gamma', mu=0, sigma=0.5, shape=3)
        delta = pm.Normal('delta', mu=0, sigma=0.5, shape=3)
        seasonality = pm.math.sum(gamma * sin_matrix + delta * cos_matrix, axis=1)

        # --- 2. DYNAMIC BASELINE SWITCHBOARD ---
        # Default to polynomial if the key is missing in the JSON
        baseline_type = config.get("baseline_type", "polynomial") 
        
        if baseline_type == "polynomial":
            beta_temp = pm.Normal('beta_temp', mu=0, sigma=0.5)
            beta_temp_sq = pm.HalfNormal('beta_temp_sq', sigma=0.5)
            weather_effect = (beta_temp * df['temp_scaled']) + (beta_temp_sq * df['temp_scaled']**2)
            
        elif baseline_type == "piecewise":
            # Using the academic Heating/Cooling degrees
            beta_heating = pm.HalfNormal('beta_heating', sigma=0.5)
            beta_cooling = pm.HalfNormal('beta_cooling', sigma=0.5)
            weather_effect = (beta_heating * df['heating_scaled']) + (beta_cooling * df['cooling_scaled'])
            
        elif baseline_type == "autoregressive":
            # Uses standard weather, but adds last week's exact behavior
            beta_temp = pm.Normal('beta_temp', mu=0, sigma=0.5)
            beta_temp_sq = pm.HalfNormal('beta_temp_sq', sigma=0.5)
            rho_lag = pm.Normal('rho_lag', mu=0.8, sigma=0.2) # High expectation that today matches last week
            weather_effect = (beta_temp * df['temp_scaled']) + (beta_temp_sq * df['temp_scaled']**2) + (rho_lag * df['demand_lag_7_scaled'])

        elif baseline_type == "thi":
            # Using the Temperature-Humidity Index instead of raw temperature
            beta_thi = pm.Normal('beta_thi', mu=0, sigma=0.5)
            beta_thi_sq = pm.HalfNormal('beta_thi_sq', sigma=0.5)
            weather_effect = (beta_thi * df['thi_scaled']) + (beta_thi_sq * df['thi_scaled']**2)

        # Combine the selected weather effect with the standard constants
        baseline = alpha + weather_effect + (beta_weekend * df['is_weekend']) + seasonality
        # 2. LATENT STATE (Hidden Solar)
        if config["latent_solar"]["dist"] == "HalfNormal":
            mu_solar = pm.HalfNormal('mu_solar', sigma=config["latent_solar"]["sigma"])
            
        elif config["latent_solar"]["dist"] == "Exponential":
            mu_solar = pm.Exponential('mu_solar', lam=config["latent_solar"]["lam"])
            
        elif config["latent_solar"]["dist"] == "LogNormal":
            # PyMC LogNormal takes mu and sigma of the underlying normal distribution
            mu_solar = pm.LogNormal('mu_solar', mu=config["latent_solar"]["mu"], sigma=config["latent_solar"]["sigma"])
            
        elif config["latent_solar"]["dist"] == "HalfCauchy":
            # HalfCauchy uses 'beta' as its scale parameter
            mu_solar = pm.HalfCauchy('mu_solar', beta=config["latent_solar"]["beta"])
            
        expected_demand = baseline - (mu_solar * df['proxy_scaled'])

        # 3. LIKELIHOOD FUNCTION
        if config["likelihood"]["dist"] == "StudentT":
            sigma_err = pm.HalfNormal('sigma_err', sigma=config["likelihood"]["sigma_err"])
            nu = pm.Exponential('nu', 1/29)
            pm.StudentT('obs', nu=nu, mu=expected_demand, sigma=sigma_err, observed=df['demand_scaled'])
            
        elif config["likelihood"]["dist"] == "Normal":
            sigma_err = pm.HalfNormal('sigma_err', sigma=config["likelihood"]["sigma_err"])
            pm.Normal('obs', mu=expected_demand, sigma=sigma_err, observed=df['demand_scaled'])
            
    return model