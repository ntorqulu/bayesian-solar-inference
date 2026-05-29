# solar_models.py
import pymc as pm
import numpy as np


def build_dynamic_model(df, config):
    """
    Builds a PyMC model dynamically from a JSON configuration entry.

    All models share:
      - Intercept, weekend effect, Fourier seasonality (K=3)
      - 46 monthly shock dummies (March 2020 - December 2023)
      - proxy_norm = selfcons_theoretical_mwh / demand_std  (non-negative, pre-2020 = 0)

    What varies per config:
      - baseline_type : 'polynomial' | 'piecewise' | 'autoregressive' | 'thi'
      - latent_solar  : prior on eta (HalfNormal, Exponential, LogNormal, HalfCauchy)
      - likelihood    : 'StudentT' (robust) | 'Normal'

    Paper references for academic baselines
    ----------------------------------------
    piecewise      : Engle et al. (1986); Bessec & Fouquau (2008)
    autoregressive : Elamin & Fukushima (2018)
    thi            : Valor et al. (2001); Mirasgedis et al. (2007)
    """
    shock_cols   = [c for c in df.columns if c.startswith('shock_')]
    shock_matrix = df[shock_cols].values if shock_cols else None

    with pm.Model() as model:

        # -- 1. SHARED PRIORS --------------------------------------------------
        alpha        = pm.Normal('alpha',        mu=0,  sigma=0.1)
        beta_weekend = pm.Normal('beta_weekend', mu=-1, sigma=0.3)

        sin_matrix  = df[['sin_k1', 'sin_k2', 'sin_k3']].values
        cos_matrix  = df[['cos_k1', 'cos_k2', 'cos_k3']].values
        gamma       = pm.Normal('gamma', mu=0, sigma=0.5, shape=3)
        delta       = pm.Normal('delta', mu=0, sigma=0.5, shape=3)
        seasonality = pm.math.sum(gamma * sin_matrix + delta * cos_matrix, axis=1)

        if shock_matrix is not None:
            beta_shocks  = pm.Normal('beta_shocks', mu=0, sigma=0.28,
                                     shape=shock_matrix.shape[1])
            shock_effect = pm.math.dot(shock_matrix, beta_shocks)
        else:
            shock_effect = 0.0

        # -- 2. BASELINE TEMPERATURE REPRESENTATION ----------------------------
        baseline_type = config.get("baseline_type", "polynomial")

        if baseline_type == "polynomial":
            # Quadratic polynomial: standard approach.
            # HalfNormal on beta_temp_sq enforces U-shaped demand curve
            # (extreme temperatures must raise demand).
            beta_temp    = pm.Normal('beta_temp',    mu=0, sigma=0.5)
            beta_temp_sq = pm.HalfNormal('beta_temp_sq',   sigma=0.5)
            weather_effect = (beta_temp    * df['temp_scaled'].values
                            + beta_temp_sq * df['temp_scaled'].values ** 2)

        elif baseline_type == "piecewise":
            # Heating Degree Days / Cooling Degree Days.
            # Based on Engle et al. (1986) and Bessec & Fouquau (2008).
            # HDD = max(18 - T, 0); CDD = max(T - 22, 0).
            # HalfNormal enforces both are strictly positive (demand rises
            # away from the 18-22 C comfort band in either direction).
            beta_heating = pm.HalfNormal('beta_heating', sigma=0.5)
            beta_cooling = pm.HalfNormal('beta_cooling', sigma=0.5)
            weather_effect = (beta_heating * df['heating_scaled'].values
                            + beta_cooling * df['cooling_scaled'].values)

        elif baseline_type == "autoregressive":
            # Polynomial weather + 7-day lagged demand.
            # Based on Elamin & Fukushima (2018) short-term load forecasting.
            # rho_lag ~ N(0.8, 0.2): strong prior that today resembles last week
            # (weekly seasonality in electricity demand is well-documented).
            # WARNING: including lagged demand as a predictor makes the model
            # circular for identification of eta. LOO score is inflated by
            # autocorrelation, not by better solar identification.
            # This model is run separately and excluded from the main LOO table.
            beta_temp    = pm.Normal('beta_temp',    mu=0, sigma=0.5)
            beta_temp_sq = pm.HalfNormal('beta_temp_sq',   sigma=0.5)
            rho_lag      = pm.Normal('rho_lag',      mu=0.8, sigma=0.2)
            weather_effect = (beta_temp    * df['temp_scaled'].values
                            + beta_temp_sq * df['temp_scaled'].values ** 2
                            + rho_lag      * df['demand_lag_7_scaled'].values)

        elif baseline_type == "thi":
            # Temperature-Humidity Index: THI = T - 0.55*(1 - RH/100)*(T - 14.5)
            # Based on Valor et al. (2001) and Mirasgedis et al. (2007), who showed
            # THI outperforms raw temperature for Spanish/Mediterranean demand.
            # Same quadratic structure as polynomial but on comfort-adjusted T.
            beta_thi    = pm.Normal('beta_thi',    mu=0, sigma=0.5)
            beta_thi_sq = pm.HalfNormal('beta_thi_sq',   sigma=0.5)
            weather_effect = (beta_thi    * df['thi_scaled'].values
                            + beta_thi_sq * df['thi_scaled'].values ** 2)

        else:
            raise ValueError(f"Unknown baseline_type: '{baseline_type}'. "
                             f"Choose 'polynomial', 'piecewise', 'autoregressive', or 'thi'.")

        baseline = (alpha
                  + weather_effect
                  + beta_weekend * df['is_weekend'].values
                  + seasonality
                  + shock_effect)

        # -- 3. LATENT SOLAR (eta) ---------------------------------------------
        # All prior families enforce eta > 0 (solar can only reduce apparent demand).
        dist = config["latent_solar"]["dist"]

        if dist == "HalfNormal":
            mu_solar = pm.HalfNormal('mu_solar',
                                     sigma=config["latent_solar"]["sigma"])
        elif dist == "Exponential":
            mu_solar = pm.Exponential('mu_solar',
                                      lam=config["latent_solar"]["lam"])
        elif dist == "LogNormal":
            mu_solar = pm.LogNormal('mu_solar',
                                    mu=config["latent_solar"]["mu"],
                                    sigma=config["latent_solar"]["sigma"])
        elif dist == "HalfCauchy":
            mu_solar = pm.HalfCauchy('mu_solar',
                                     beta=config["latent_solar"]["beta"])
        else:
            raise ValueError(f"Unknown latent_solar dist: '{dist}'.")

        expected_demand = baseline - (mu_solar * df['proxy_norm'].values)

        # -- 4. LIKELIHOOD -----------------------------------------------------
        sigma_err = pm.HalfNormal('sigma_err',
                                  sigma=config["likelihood"]["sigma_err"])
        lik = config["likelihood"]["dist"]

        if lik == "StudentT":
            nu = pm.Exponential('nu', 1 / 29)
            pm.StudentT('obs', nu=nu, mu=expected_demand, sigma=sigma_err,
                        observed=df['demand_scaled'].values)
        elif lik == "Normal":
            pm.Normal('obs', mu=expected_demand, sigma=sigma_err,
                      observed=df['demand_scaled'].values)
        else:
            raise ValueError(f"Unknown likelihood dist: '{lik}'.")

    return model
