# Global Shark Attack Risk Heatmap Generator
import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Try to import cartopy for world map features
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Note: cartopy not available, using basic matplotlib projection")

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
    
    # Apply temperature adjustment to shark density (same as in training)
    def apply_temperature_adjustment(row):
        shark_density = row['Real_Shark_Density']
        sst = row['SST_Celsius']
        year = row['Year']
        
        # Add climate change factor: 0.03°C increase per year since 2010
        climate_adjusted_sst = sst + (year - 2010) * 0.03
        
        # Temperature-based shark activity factor (same as training)
        if 26 <= climate_adjusted_sst <= 28:
            # Optimal temperature range - increase activity
            temp_factor = 1.5
        elif 24 <= climate_adjusted_sst < 26 or 28 < climate_adjusted_sst <= 30:
            # Good temperature range
            temp_factor = 1.2
        elif 20 <= climate_adjusted_sst < 24 or 30 < climate_adjusted_sst <= 32:
            # Acceptable range
            temp_factor = 1.0
        elif 16 <= climate_adjusted_sst < 20 or 32 < climate_adjusted_sst <= 34:
            # Suboptimal - less activity
            temp_factor = 0.8
        elif climate_adjusted_sst < 16:
            # Too cold - avoid or sluggish
            temp_factor = 0.4
        else:  # climate_adjusted_sst > 34
            # Too hot - seek cooler water
            temp_factor = 0.6
        
        return shark_density * temp_factor
    
    df['Temp_Adjusted_Shark_Density'] = df.apply(apply_temperature_adjustment, axis=1)
    
    # Add log population (as used in training)
    df['Log_Population'] = np.log1p(df['Population'])
    
    # Select features in the same order as training
    X = df[feature_names].copy()
    
    print(f"Predicting for {len(X):,} locations...")
    
    # Get predictions from model (no post-processing temperature adjustments needed)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Simple constraint: reduce probability for zero shark areas + apply climate multiplier
    constrained_probabilities = []
    for i, prob in enumerate(probabilities):
        temp_adjusted_shark_density = df.iloc[i]['Temp_Adjusted_Shark_Density']
        year = df.iloc[i]['Year']
        sst = df.iloc[i]['SST_Celsius']
        
        if temp_adjusted_shark_density == 0:
            # If no temperature-adjusted sharks present, reduce probability
            constrained_prob = prob * 0.01  # Reduce by 99%
        else:
            # For areas with sharks, use model prediction as-is
            # (temperature effects already learned during training)
            constrained_prob = prob
            
            # Additional post-processing: Apply climate change multiplier for more dramatic effect
            climate_adjusted_sst = sst + (year - 2010) * 0.03
            
            # More aggressive climate-based probability multiplier
            if climate_adjusted_sst > 28:
                # Waters getting too hot - potential stress/displacement
                climate_multiplier = 0.7 + (climate_adjusted_sst - 28) * 0.1  # Increases risk paradoxically due to stress
            elif climate_adjusted_sst > 26:
                # Optimal warming - increased activity
                climate_multiplier = 1.0 + (climate_adjusted_sst - 26) * 0.3  # Up to 1.6x at 28°C
            elif climate_adjusted_sst > 20:
                # Moderate warming - some increase
                climate_multiplier = 1.0 + (climate_adjusted_sst - 20) * 0.05  # Up to 1.3x at 26°C
            else:
                # Cooler waters - baseline
                climate_multiplier = 1.0
                
            constrained_prob = min(constrained_prob * climate_multiplier, 0.999)  # Cap at 99.9%
        
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
    
    # Temperature-adjusted shark density analysis
    zero_temp_shark_mask = df['Temp_Adjusted_Shark_Density'] == 0
    zero_temp_shark_count = zero_temp_shark_mask.sum()
    zero_temp_shark_prob_mean = probabilities[zero_temp_shark_mask].mean()
    
    print(f"\n🦈 Temperature-Adjusted Shark Density Analysis:")
    print(f"  Locations with zero temp-adjusted sharks: {zero_temp_shark_count:,} ({zero_temp_shark_count/len(df):.1%})")
    print(f"  Mean probability at zero temp-adjusted shark areas: {zero_temp_shark_prob_mean:.1%}")
    
    # Compare raw vs temperature-adjusted shark density
    zero_raw_shark_mask = df['Real_Shark_Density'] == 0
    zero_raw_shark_count = zero_raw_shark_mask.sum()
    print(f"  Locations with zero raw sharks: {zero_raw_shark_count:,} ({zero_raw_shark_count/len(df):.1%})")
    print(f"  Temperature adjustment effect: {zero_raw_shark_count - zero_temp_shark_count:,} locations gained/lost sharks")
    
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

def create_global_heatmap(lat_range, lon_range, prob_grid, year, month, 
                         output_file="maps/global_shark_attack_risk.png"):
    """Create a global heatmap of shark attack probabilities with world map features."""
    print(f"Creating global shark attack risk heatmap...")
    
    if HAS_CARTOPY:
        # Use cartopy for proper world map projection
        fig = plt.figure(figsize=(24, 14))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray', alpha=0.7)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.2)
        
        # Create heatmap
        heatmap = ax.imshow(prob_grid, 
                           extent=[lon_range.min(), lon_range.max(), 
                                  lat_range.min(), lat_range.max()],
                           origin='lower', 
                           cmap='YlOrRd', 
                           transform=ccrs.PlateCarree(),
                           alpha=0.8,
                           interpolation='bilinear')
        
        # Set global extent
        ax.set_global()
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        
    else:
        # Fallback to matplotlib with enhanced features
        fig, ax = plt.subplots(figsize=(24, 14))
        
        # Create heatmap
        heatmap = ax.imshow(prob_grid, 
                           extent=[lon_range.min(), lon_range.max(), 
                                  lat_range.min(), lat_range.max()],
                           origin='lower', 
                           cmap='YlOrRd', 
                           aspect='auto',
                           alpha=0.8,
                           interpolation='bilinear')
        
        # Add enhanced grid and reference lines
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add major geographic reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1.2, label='Equator')
        ax.axhline(y=23.5, color='blue', linestyle='--', alpha=0.7, linewidth=1, label='Tropic of Cancer')
        ax.axhline(y=-23.5, color='blue', linestyle='--', alpha=0.7, linewidth=1, label='Tropic of Capricorn')
        ax.axhline(y=66.5, color='purple', linestyle=':', alpha=0.7, linewidth=1, label='Arctic Circle')
        ax.axhline(y=-66.5, color='purple', linestyle=':', alpha=0.7, linewidth=1, label='Antarctic Circle')
        
        # Add major meridians
        for lon in [-180, -120, -60, 0, 60, 120, 180]:
            ax.axvline(x=lon, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
        
        # Set labels and limits
        ax.set_xlabel('Longitude', fontsize=16, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=16, fontweight='bold')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        
        # Add a small legend for reference lines
        ax.legend(loc='upper left', fontsize=10, framealpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Shark Attack Probability', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    
    # Format colorbar as percentages
    cbar_ticks = cbar.get_ticks()
    cbar.ax.set_yticklabels([f'{tick:.0%}' for tick in cbar_ticks])
    
    # Add title
    plt.suptitle(f'Global Shark Attack Risk Heatmap - {year}/{month:02d}', 
                fontsize=22, fontweight='bold', y=0.95)
    
    # Add subtitle with model info
    plt.figtext(0.5, 0.91, 
               'Based on XGBoost ML model using real shark density, temperature, population, and geographic factors',
               ha='center', fontsize=14, style='italic', alpha=0.8)
    
    # Find and annotate highest risk location
    max_prob_idx = np.unravel_index(np.argmax(prob_grid), prob_grid.shape)
    max_lat = lat_range[max_prob_idx[0]]
    max_lon = lon_range[max_prob_idx[1]]
    max_prob = prob_grid[max_prob_idx]
    
    # Only annotate if it's a significant risk and visible
    if max_prob > 0.3:
        if HAS_CARTOPY:
            ax.annotate(f'Highest Risk\n{max_prob:.0%}', 
                       xy=(max_lon, max_lat), 
                       xytext=(max_lon + 15, max_lat + 15),
                       transform=ccrs.PlateCarree(),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='red'))
        else:
            ax.annotate(f'Highest Risk\n{max_prob:.0%}', 
                       xy=(max_lon, max_lat), 
                       xytext=(max_lon + 15, max_lat + 15),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='red'))
    
    # Style the plot
    if not HAS_CARTOPY:
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Save the plot
    output_dir = Path(output_file).parent
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"✓ Enhanced world map heatmap saved to {output_file}")
    plt.show()
    
    return max_lat, max_lon, max_prob

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

def main_prediction_pipeline(num_samples=90000, year=2024, month=7, sigma=2.0):
    """Complete pipeline to generate global shark attack risk heatmap."""
    print(f"🌍 GLOBAL SHARK ATTACK RISK ANALYSIS")
    print(f"📅 Date: {year}/{month:02d}")
    print(f"📍 Samples: {num_samples:,} locations worldwide")
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
    
    # Step 5: Create heatmap
    print("5️⃣ Creating global heatmap...")
    max_lat, max_lon, max_prob = create_global_heatmap(lat_range, lon_range, prob_grid, year, month)
    
    # Step 6: Generate summary
    print("6️⃣ Generating risk summary...")
    generate_risk_summary(results, year, month)
    
    print(f"\n🎉 Analysis complete! Check 'maps/global_shark_attack_risk.png' for the heatmap.")
    
    return results, lat_range, lon_range, prob_grid

if __name__ == "__main__":
    # Run full pipeline to generate global heatmap with 90,000 uniform points
    print("🌍 STARTING GLOBAL SHARK ATTACK RISK ANALYSIS")
    print("="*60)
    
    results, lat_range, lon_range, prob_grid = main_prediction_pipeline(
        num_samples=1000000,  # High resolution: 300x300 grid (1.2° resolution)
        year=2024, 
        month=7,  # July (summer in Northern Hemisphere)
        sigma=7  # Smoothing parameter
    )

    results, lat_range, lon_range, prob_grid = main_prediction_pipeline(
        num_samples=1000000,  # High resolution: 300x300 grid (1.2° resolution)
        year=2024, 
        month=12,  # December (winter in Northern Hemisphere)
        sigma=7  # Smoothing parameter
    )




