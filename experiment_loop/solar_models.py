# solar_models.py
import pymc as pm
import numpy as np

def build_dynamic_model(df, config):
    """
    Builds a PyMC model dynamically based on a loaded JSON configuration.
    """
    with pm.Model() as model:
        
        # 1. BASELINE PRIORS
        alpha = pm.Normal('alpha', mu=0, sigma=0.1)
        beta_weekend = pm.Normal('beta_weekend', mu=-1, sigma=0.3)
        
        if config["baseline_temp"]["dist"] == "Normal":
            beta_temp = pm.Normal('beta_temp', mu=0, sigma=config["baseline_temp"]["sigma"])
            beta_temp_sq = pm.HalfNormal('beta_temp_sq', sigma=config["baseline_temp"]["sigma"])
        
        # Seasonality
        sin_matrix = df[['sin_k1', 'sin_k2', 'sin_k3']].values
        cos_matrix = df[['cos_k1', 'cos_k2', 'cos_k3']].values
        gamma = pm.Normal('gamma', mu=0, sigma=0.5, shape=3)
        delta = pm.Normal('delta', mu=0, sigma=0.5, shape=3)
        seasonality = pm.math.sum(gamma * sin_matrix + delta * cos_matrix, axis=1)

        baseline = alpha + (beta_temp * df['temp_scaled']) + (beta_temp_sq * df['temp_scaled']**2) + (beta_weekend * df['is_weekend']) + seasonality

        # 2. LATENT STATE (Hidden Solar)
        if config["latent_solar"]["dist"] == "HalfNormal":
            mu_solar = pm.HalfNormal('mu_solar', sigma=config["latent_solar"]["sigma"])
        elif config["latent_solar"]["dist"] == "Exponential":
            mu_solar = pm.Exponential('mu_solar', lam=config["latent_solar"]["lam"])
            
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