# Global Shark Attack Risk Data Generator for Webapp
# Simplified version of prediction.py that just saves data without plotting
# Author: GitHub Copilot

import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from pathlib import Path
import joblib
from scipy.ndimage import gaussian_filter
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def generate_uniform_samples(num_samples, year, month):
    """Generate uniformly distributed samples across the globe for a given date."""
    samples = []
    num_each = int(np.sqrt(num_samples))
    for i in range(num_each):
        for j in range(num_each):
            lon = -180 + (360 / num_each) * i
            lat = -90 + (180 / num_each) * j
            samples.append([year, month, round(lon, 6), round(lat, 6)])
    return samples

def calcualte_temp(samples):
    for sample in samples:
        year = sample[0]
        month = sample[1]
        lon = sample[2]
        lat = sample[3]

        # Base temperature from latitude (warmer at equator)
        base_temp = 28 - abs(lat) * 0.3
        
        # Seasonal variation
        if lat >= 0:  # Northern hemisphere
            seasonal = 4 * np.cos((month - 8) * np.pi / 6)  # Peak in August
        else:  # Southern hemisphere
            seasonal = 4 * np.cos((month - 2) * np.pi / 6)  # Peak in February
        
        # Ocean basin effects
        basin_effect = 0
        if -60 <= lon <= 20:  # Atlantic
            basin_effect = 1
        elif 100 <= lon <= 180:  # Western Pacific (warmer)
            basin_effect = 2
        elif -180 <= lon <= -60:  # Eastern Pacific (cooler)
            basin_effect = -1
        
        # Upwelling zones (cooler)
        upwelling = 0
        if (lat > 0 and ((lon >= -130 and lon <= -110) or  # California
                        (lon >= -20 and lon <= 10))):      # Canary
            upwelling = -3
        elif (lat < 0 and ((lon >= -90 and lon <= -70) or  # Peru
                        (lon >= 10 and lon <= 20))):     # Benguela
            upwelling = -3
        
        # Final SST
        sst = base_temp + seasonal + basin_effect + upwelling
        sst += (year - 2010) * 0.03

        # Realistic bounds
        sst = max(-2, min(32, sst))
        
        sst = round(sst, 1)
        sample.append(sst)
    return samples

def calculate_population(samples, country_file="data/population/country-capital-lat-long-population.csv"):
    """Calculate population for each sample using vectorized distance calculations for speed."""
    import numpy as np
    
    # Load country data
    country_data = pd.read_csv(country_file)
    print(f"Calculating population for {len(samples):,} samples using vectorized approach...")
    
    # Convert country coordinates to numpy arrays for vectorized operations
    country_lats = country_data['Latitude'].values
    country_lons = country_data['Longitude'].values
    country_pops = country_data['Population'].values
    
    def find_closest_country_vectorized(lat, lon):
        """Find closest country using vectorized numpy operations - much faster!"""
        # Calculate all distances at once using the haversine approximation
        # For speed, we'll use simple Euclidean distance with latitude scaling
        lat_diff = country_lats - lat
        lon_diff = (country_lons - lon) * np.cos(np.radians(lat))  # Scale longitude by latitude
        
        # Simple distance approximation (much faster than geodesic)
        distances = np.sqrt(lat_diff**2 + lon_diff**2)
        
        # Find the index of minimum distance
        closest_idx = np.argmin(distances)
        return country_pops[closest_idx]
    
    # Add population to each sample with progress tracking
    for i, sample in enumerate(samples):
        if i % 1000 == 0:  # Progress indicator
            print(f"  Progress: {i:,}/{len(samples):,} ({i/len(samples)*100:.1f}%)")
            
        lon, lat = sample[2], sample[3]
        population = find_closest_country_vectorized(lat, lon)
        sample.append(population)
    
    print(f"✓ Population calculation complete!")
    return samples

def calculate_shark_density_from_table(samples, shark_density_file="data/processed/shark_density_grid.csv"):
    """Calculate shark density for each sample using fast dictionary lookup."""
    print(f"Loading shark density grid from {shark_density_file}...")
    
    # Load shark density grid once
    density_grid = pd.read_csv(shark_density_file)
    
    # Create a dictionary for O(1) lookup instead of O(n) pandas search
    print("Creating fast lookup dictionary...")
    density_dict = {}
    for _, row in density_grid.iterrows():
        key = (row['LatBin'], row['LonBin'])
        density_dict[key] = row['NormalizedDensity']
    
    print(f"Calculating shark density for {len(samples):,} samples...")
    
    # Add shark density to each sample using fast dictionary lookup
    for i, sample in enumerate(samples):
        if i % 1000 == 0:  # Progress indicator
            print(f"  Progress: {i:,}/{len(samples):,} ({i/len(samples)*100:.1f}%)")
            
        lon, lat = sample[2], sample[3]
        
        # Round to nearest 0.5° grid cell
        lat_bin = round(lat / 0.5) * 0.5
        lon_bin = round(lon / 0.5) * 0.5
        
        # Fast dictionary lookup - O(1) instead of O(n)
        shark_density = density_dict.get((lat_bin, lon_bin), 0.0)
        sample.append(shark_density)
    
    print(f"✓ Shark density calculation complete!")
    return samples

def predict_shark_attack_probabilities(samples, model_path="models/shark_attack_xgboost_environmental_model.pkl"):
    """Use trained XGBoost model to predict shark attack probabilities for all samples."""
    print(f"Loading trained model from {model_path}...")
    
    # Load the trained model
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    print(f"Model features: {feature_names}")
    
    # Convert samples to DataFrame with proper feature names
    df = pd.DataFrame(samples, columns=[
        'Year', 'Month', 'Longitude', 'Latitude', 
        'SST_Celsius', 'Population', 'Real_Shark_Density'
    ])
    
    # Add log population (as used in training)
    df['Log_Population'] = np.log1p(df['Population'])
    
    # Select features in the same order as training
    X = df[feature_names].copy()
    
    print(f"Predicting for {len(X):,} locations...")
    
    # Get raw predictions from model
    raw_probabilities = model.predict_proba(X)[:, 1]
    
    # Apply post-processing constraints based on shark density and climate-adjusted temperature
    constrained_probabilities = []
    for i, raw_prob in enumerate(raw_probabilities):
        shark_density = df.iloc[i]['Real_Shark_Density']
        sst = df.iloc[i]['SST_Celsius']
        year = df.iloc[i]['Year']
        
        # Apply same constraints as in training model
        if shark_density == 0:
            # If no sharks present, reduce probability dramatically
            constrained_prob = raw_prob * 0.01  # Reduce by 99%
        elif shark_density < 0.05:  # Very low shark density
            # Scale down probability proportionally
            scaling_factor = shark_density / 0.05  # Linear scaling from 0 to 1
            constrained_prob = raw_prob * (0.01 + 0.99 * scaling_factor)
        else:
            constrained_prob = raw_prob
        
        # Apply climate-adjusted temperature scaling (post-processing effect)
        # Add climate change factor: 0.03°C increase per year since 2010
        climate_adjusted_sst = sst + (year - 2010) * 0.03
        
        # Temperature-based probability multiplier
        if climate_adjusted_sst >= 28:
            # Very warm waters - high shark activity
            temp_multiplier = 1.4
        elif climate_adjusted_sst >= 26:
            # Optimal temperature range - increased activity
            temp_multiplier = 1.3
        elif climate_adjusted_sst >= 24:
            # Good temperature range
            temp_multiplier = 1.1
        elif climate_adjusted_sst >= 20:
            # Acceptable range
            temp_multiplier = 1.0
        elif climate_adjusted_sst >= 16:
            # Suboptimal - reduced activity
            temp_multiplier = 0.8
        else:
            # Too cold - very low activity
            temp_multiplier = 0.6
        
        # Apply temperature scaling
        constrained_prob = constrained_prob * temp_multiplier
        
        # Ensure probability doesn't exceed reasonable bounds
        constrained_prob = min(constrained_prob, 0.95)  # Cap at 95%
        constrained_probabilities.append(constrained_prob)
    
    probabilities = np.array(constrained_probabilities)
    
    print(f"\n📊 Global Prediction Analysis:")
    print(f"  Mean probability: {probabilities.mean():.1%}")
    print(f"  Std probability:  {probabilities.std():.1%}")
    print(f"  Max probability:  {probabilities.max():.1%}")
    print(f"  Min probability:  {probabilities.min():.1%}")
    
    # Climate change impact analysis
    year = df['Year'].iloc[0]  # Assuming single year prediction
    base_sst_mean = df['SST_Celsius'].mean()
    climate_adjustment = (year - 2010) * 0.03
    climate_adjusted_sst_mean = base_sst_mean + climate_adjustment
    
    print(f"\n🌡️ Climate Change Impact Analysis (Year {year}):")
    print(f"  Base SST mean: {base_sst_mean:.2f}°C")
    print(f"  Climate adjustment: +{climate_adjustment:.2f}°C")
    print(f"  Climate-adjusted SST mean: {climate_adjusted_sst_mean:.2f}°C")
    print(f"  Years since baseline (2010): {year - 2010}")
    
    # Show effect of post-processing
    raw_mean = raw_probabilities.mean()
    constrained_mean = probabilities.mean()
    print(f"\n🔧 Post-processing Impact:")
    print(f"  Raw model mean:        {raw_mean:.1%}")
    print(f"  Constrained mean:      {constrained_mean:.1%}")
    print(f"  Reduction factor:      {raw_mean/constrained_mean:.1f}x")
    
    # Zero shark density analysis
    zero_shark_mask = df['Real_Shark_Density'] == 0
    zero_shark_count = zero_shark_mask.sum()
    zero_shark_prob_mean = probabilities[zero_shark_mask].mean()
    
    print(f"\n🦈 Shark Density Analysis:")
    print(f"  Locations with zero sharks: {zero_shark_count:,} ({zero_shark_count/len(df):.1%})")
    print(f"  Mean probability at zero shark areas: {zero_shark_prob_mean:.1%}")
    
    # Analyze probability distribution
    low_risk = (probabilities < 0.1).sum()
    med_risk = ((probabilities >= 0.1) & (probabilities < 0.5)).sum()
    high_risk = (probabilities >= 0.5).sum()
    
    print(f"\n📈 Risk Distribution:")
    print(f"  Low risk (<10%):     {low_risk:5d} ({low_risk/len(probabilities):.1%})")
    print(f"  Medium risk (10-50%): {med_risk:5d} ({med_risk/len(probabilities):.1%})")
    print(f"  High risk (≥50%):    {high_risk:5d} ({high_risk/len(probabilities):.1%})")
    
    # Add coordinates and probabilities for mapping
    results = []
    for i, prob in enumerate(probabilities):
        lat, lon = samples[i][3], samples[i][2]  # lat, lon from samples
        results.append([lat, lon, prob])
    
    print(f"✓ Predictions complete!")
    
    return results

def smooth_predictions(results, sigma=1.0):
    """Smooth the predictions using Gaussian filter for better visualization."""
    print(f"Smoothing predictions with sigma={sigma}...")
    
    # Convert to arrays
    lats = np.array([r[0] for r in results])
    lons = np.array([r[1] for r in results])
    probs = np.array([r[2] for r in results])
    
    # Create regular grid
    lat_range = np.linspace(lats.min(), lats.max(), int(np.sqrt(len(results))))
    lon_range = np.linspace(lons.min(), lons.max(), int(np.sqrt(len(results))))
    
    # FIXED: Reshape probabilities to grid correctly
    # Our samples are generated with longitude as outer loop, latitude as inner loop
    # So we need to reshape as (n_lon, n_lat) then transpose to get (n_lat, n_lon)
    prob_grid = probs.reshape(len(lon_range), len(lat_range)).T
    
    # Apply Gaussian smoothing
    smoothed_grid = gaussian_filter(prob_grid, sigma=sigma)
    
    print("✓ Smoothing complete!")
    
    return lat_range, lon_range, smoothed_grid

def generate_risk_summary(results, year, month):
    """Generate a summary of risk statistics."""
    probs = np.array([r[2] for r in results])
    
    print(f"\n🦈 GLOBAL SHARK ATTACK RISK SUMMARY - {year}/{month:02d}")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"  • Total locations analyzed: {len(results):,}")
    print(f"  • Average risk globally: {probs.mean():.2%}")
    print(f"  • Maximum risk found: {probs.max():.2%}")
    print(f"  • Minimum risk found: {probs.min():.2%}")
    print(f"  • Standard deviation: {probs.std():.2%}")
    
    # Risk categories
    high_risk = (probs >= 0.5).sum()
    medium_risk = ((probs >= 0.1) & (probs < 0.5)).sum()
    low_risk = (probs < 0.1).sum()
    
    print(f"\n🎯 Risk Categories:")
    print(f"  • High Risk (≥50%): {high_risk:,} locations ({high_risk/len(results):.1%})")
    print(f"  • Medium Risk (10-50%): {medium_risk:,} locations ({medium_risk/len(results):.1%})")
    print(f"  • Low Risk (<10%): {low_risk:,} locations ({low_risk/len(results):.1%})")
    
    # Find top 5 highest risk locations
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    print(f"\n🏆 Top 5 Highest Risk Locations:")
    for i, (lat, lon, prob) in enumerate(sorted_results[:5], 1):
        print(f"  {i}. Lat: {lat:6.2f}°, Lon: {lon:7.2f}° - Risk: {prob:.1%}")

def generate_webapp_data_for_date(year, month, num_samples=40000, sigma=2.0):
    """Generate and save data for a specific date for webapp usage."""
    print(f"\n🌍 GENERATING WEBAPP DATA FOR {year}/{month:02d}")
    print("=" * 60)
    
    # Step 1: Generate samples
    print("\n1️⃣ Generating uniform global samples...")
    samples = generate_uniform_samples(num_samples, year, month)
    
    # Step 2: Calculate features
    print("2️⃣ Calculating environmental features...")
    samples = calcualte_temp(samples)
    samples = calculate_population(samples)
    samples = calculate_shark_density_from_table(samples)
    
    # Step 3: Predict probabilities
    print("3️⃣ Predicting shark attack probabilities...")
    results = predict_shark_attack_probabilities(samples)
    
    # Step 4: Smooth predictions
    print("4️⃣ Smoothing predictions for visualization...")
    lat_range, lon_range, prob_grid = smooth_predictions(results, sigma=sigma)
    
    # Step 5: Generate summary
    print("5️⃣ Generating risk summary...")
    generate_risk_summary(results, year, month)
    
    # Step 6: Prepare data for webapp
    probabilities = np.array([r[2] for r in results])
    
    # Calculate statistics
    stats = {
        'mean_probability': float(probabilities.mean()),
        'max_probability': float(probabilities.max()),
        'min_probability': float(probabilities.min()),
        'std_probability': float(probabilities.std()),
        'high_risk_count': int((probabilities >= 0.5).sum()),
        'medium_risk_count': int(((probabilities >= 0.1) & (probabilities < 0.5)).sum()),
        'low_risk_count': int((probabilities < 0.1).sum()),
        'total_locations': len(probabilities)
    }
    
    # Climate change analysis
    base_sst_mean = np.mean([s[4] for s in samples])  # SST is at index 4
    climate_adjustment = (year - 2010) * 0.03
    climate_adjusted_sst_mean = base_sst_mean + climate_adjustment
    
    climate_info = {
        'base_sst_mean': float(base_sst_mean),
        'climate_adjustment': float(climate_adjustment),
        'climate_adjusted_sst_mean': float(climate_adjusted_sst_mean),
        'years_since_baseline': year - 2010
    }
    
    # Find top risk locations
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    top_locations = [
        {
            'lat': float(r[0]),
            'lon': float(r[1]), 
            'risk': float(r[2])
        } for r in sorted_results[:10]
    ]
    
    # Create output data structure
    webapp_data = {
        'year': year,
        'month': month,
        'date_string': f"{year}-{month:02d}",
        'lat_range': lat_range.tolist(),
        'lon_range': lon_range.tolist(),
        'prob_grid': prob_grid.tolist(),
        'statistics': stats,
        'climate_info': climate_info,
        'top_locations': top_locations,
        'grid_resolution': {
            'lat_points': len(lat_range),
            'lon_points': len(lon_range)
        },
        'generated_at': datetime.now().isoformat()
    }
    
    # Save to JSON file
    output_dir = Path("public/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"heatmap_{year}_{month:02d}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(webapp_data, f, indent=2)
    
    print(f"💾 Saved webapp data to: {filepath}")
    print(f"✅ Ready for interactive webapp!")
    
    return webapp_data

if __name__ == "__main__":
    # Generate data for a few sample dates first
    print("🚀 GENERATING SAMPLE WEBAPP DATA")
    print("=" * 60)
    
    # Generate a few sample dates for testing
    sample_dates = [
        (2020, 1),   # January 2020
        (2022, 7),   # July 2022  
        (2024, 12)   # December 2024
    ]
    
    for year, month in sample_dates:
        try:
            data = generate_webapp_data_for_date(year, month)
            print(f"✅ Successfully generated data for {year}/{month:02d}")
        except Exception as e:
            print(f"❌ Error generating data for {year}/{month:02d}: {e}")
    
    print(f"\n🎉 Sample data generation complete!")
    print(f"📁 Check public/data/ folder for JSON files")
    print(f"🌐 Ready to build React webapp!")