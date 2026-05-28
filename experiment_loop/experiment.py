import os
import json
import arviz as az
import pymc as pm
from solar_models import build_dynamic_model
import matplotlib.pyplot as plt

# 1. Load the configuration file
with open('experiments_config.json', 'r') as file:
    experiments = json.load(file)

os.makedirs('models', exist_ok=True)
results_dict = {}

# 2. Execute the Pipeline
for config in experiments:
    model_name = config["name"]
    file_path = f"models/trace_{model_name}.nc"
    
    print(f"\n--- Processing: {model_name} ---")
    
    if os.path.exists(file_path):
        print(f"Loading existing trace from {file_path}...")
        trace = az.from_netcdf(file_path)
    else:
        print(f"Building model and running MCMC for {model_name}...")
        
        # Build the model using the imported blueprint and current JSON config
        model = build_dynamic_model(df, config)
        
        with model:
            trace = pm.sample(draws=1000, 
                              tune=1000, 
                              chains=4, 
                              target_accept=0.95, 
                              nuts_sampler="numpyro", 
                              return_inferencedata=True,
                              random_seed=42)
            
            print(f"Saving to {file_path}...")
            az.to_netcdf(trace, file_path)
    
    results_dict[model_name] = trace

print("\nPipeline execution complete!")

# 1. Compare the models using Leave-One-Out (LOO) cross-validation
print("Calculating Leave-One-Out Cross-Validation (LOO) for all models...")
comparison_df = az.compare(results_dict, ic="loo", scale="deviance")

# 2. Display the mathematical leaderboard
print("\n--- BAYESIAN MODEL LEADERBOARD ---")
display(comparison_df)

# 3. Visualize the comparison
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_compare(comparison_df, ax=ax, textsize=12)
ax.set_title("Bayesian Model Comparison (Expected Predictive Accuracy)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()