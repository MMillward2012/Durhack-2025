#!/usr/bin/env python3
"""
Real Shark Density Fetcher using GBIF
=====================================

Downloads REAL shark occurrence data from GBIF (Global Biodiversity Information Facility)
and calculates actual shark density for our attack/non-attack locations.

GBIF has 3.5+ billion shark occurrence records with coordinates!

Author: GitHub Copilot  
Date: November 2025
"""

import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealSharkDensityFetcher:
    """Fetches real shark occurrence data from GBIF and calculates density."""
    
    def __init__(self):
        self.base_url = "https://api.gbif.org/v1/occurrence/search"
        self.shark_species = [
            "Carcharodon carcharias",  # Great White Shark
            "Galeocerdo cuvier",       # Tiger Shark  
            "Carcharhinus leucas",     # Bull Shark
            "Prionace glauca",         # Blue Shark
            "Isurus oxyrinchus",       # Shortfin Mako
            "Carcharhinus longimanus", # Oceanic Whitetip
            "Negaprion brevirostris",  # Lemon Shark
            "Sphyrna mokarran",        # Great Hammerhead
        ]
        
    def download_shark_occurrences(self, max_records=5000):
        """Download real shark occurrence data from GBIF (fast version)."""
        
        output_dir = Path('data/real_shark_observations')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'gbif_sharks.csv'
        
        if output_file.exists():
            logger.info(f"✅ Real shark data already exists: {output_file}")
            return output_file
            
        logger.info("🦈 Downloading REAL shark occurrence data from GBIF (FAST MODE)...")
        logger.info(f"Target: {max_records:,} records from top shark species")
        
        # Focus on just the most dangerous/common species for speed
        priority_species = [
            "Carcharodon carcharias",  # Great White Shark
            "Galeocerdo cuvier",       # Tiger Shark  
            "Carcharhinus leucas",     # Bull Shark
        ]
        
        all_records = []
        total_downloaded = 0
        
        for species in priority_species:
            if total_downloaded >= max_records:
                break
                
            logger.info(f"📥 Fetching {species}...")
            species_records = self._fetch_species_data(species, limit=min(1500, max_records - total_downloaded))
            
            if species_records:
                all_records.extend(species_records)
                total_downloaded += len(species_records)
                logger.info(f"  Downloaded {len(species_records):,} records")
            else:
                logger.warning(f"  No data for {species}")
            
            # Minimal delay
            time.sleep(0.5)
        
        if all_records:
            # Convert to DataFrame
            df = pd.DataFrame(all_records)
            
            # Clean and validate coordinates
            df = df.dropna(subset=['latitude', 'longitude'])
            df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
            
            # Remove duplicates (same lat/lon)
            df = df.drop_duplicates(subset=['latitude', 'longitude'])
            
            # Save to file
            df.to_csv(output_file, index=False)
            
            logger.info(f"✅ Downloaded {len(df):,} real shark occurrences")
            logger.info(f"📁 Saved to: {output_file}")
            
            # Show geographic spread
            lat_range = f"{df['latitude'].min():.1f} to {df['latitude'].max():.1f}"
            lon_range = f"{df['longitude'].min():.1f} to {df['longitude'].max():.1f}"
            logger.info(f"📍 Geographic coverage: Lat {lat_range}, Lon {lon_range}")
            
            return output_file
        else:
            logger.error("❌ No shark data downloaded")
            return None
    
    def _fetch_species_data(self, species_name, limit=1500):
        """Fetch occurrence data for a specific shark species (fast version)."""
        
        params = {
            'scientificName': species_name,
            'hasCoordinate': 'true',
            'limit': min(limit, 500),  # Smaller chunks for speed
            'offset': 0
        }
        
        records = []
        
        try:
            # Only do 1-3 requests per species for speed
            max_requests = 3
            requests_made = 0
            
            while len(records) < limit and requests_made < max_requests:
                response = requests.get(self.base_url, params=params, timeout=10)
                
                if response.status_code != 200:
                    logger.warning(f"API error for {species_name}: {response.status_code}")
                    break
                
                data = response.json()
                
                if not data.get('results'):
                    break
                
                # Extract key fields
                for record in data['results']:
                    if 'decimalLatitude' in record and 'decimalLongitude' in record:
                        records.append({
                            'species': species_name,
                            'latitude': float(record['decimalLatitude']),
                            'longitude': float(record['decimalLongitude']),
                            'year': record.get('year'),
                            'country': record.get('country'),
                            'dataset': record.get('datasetKey')
                        })
                
                requests_made += 1
                
                # Check if we have more data
                if data.get('endOfRecords', True):
                    break
                    
                # Next page
                params['offset'] += params['limit']
                
                # Minimal rate limiting
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error fetching {species_name}: {e}")
        
        return records
    
    def load_shark_data(self):
        """Load real shark occurrence data."""
        
        shark_file = self.download_shark_occurrences()
        if not shark_file:
            return None
            
        try:
            df = pd.read_csv(shark_file)
            logger.info(f"📊 Loaded {len(df):,} real shark occurrences")
            
            # Show species breakdown
            species_counts = df['species'].value_counts()
            logger.info("🦈 Species breakdown:")
            for species, count in species_counts.head(5).items():
                short_name = species.split()[-1]  # Last word (species name)
                logger.info(f"  {short_name}: {count:,} records")
            
            return df[['latitude', 'longitude', 'species']]
            
        except Exception as e:
            logger.error(f"❌ Error loading shark data: {e}")
            return None
    
    def calculate_real_shark_density(self, target_lat, target_lon, shark_df, radius_km=100):
        """Calculate real shark density around a target location."""
        
        if shark_df is None or len(shark_df) == 0:
            return 0.0
        
        # Calculate distance to all shark observations using Haversine approximation
        lat_diff = np.radians(shark_df['latitude'] - target_lat)
        lon_diff = np.radians(shark_df['longitude'] - target_lon)
        
        # Haversine formula (more accurate than simple degree difference)
        a = (np.sin(lat_diff/2)**2 + 
             np.cos(np.radians(target_lat)) * np.cos(np.radians(shark_df['latitude'])) * 
             np.sin(lon_diff/2)**2)
        
        distance_km = 6371 * 2 * np.arcsin(np.sqrt(a))  # Earth radius = 6371 km
        
        # Count unique species within radius (more meaningful than total observations)
        nearby_mask = distance_km <= radius_km
        nearby_sharks = shark_df[nearby_mask]
        
        if len(nearby_sharks) == 0:
            return 0.0
        
        # Calculate density metrics
        total_observations = len(nearby_sharks)
        unique_species = nearby_sharks['species'].nunique()
        
        # Area in km²
        area_km2 = np.pi * (radius_km ** 2)
        
        # Density = observations per 1000 km²
        observation_density = (total_observations / area_km2) * 1000
        
        # Species richness = unique species per 1000 km²
        species_density = (unique_species / area_km2) * 1000
        
        # Combined metric (weighted average)
        combined_density = (0.7 * observation_density + 0.3 * species_density)
        
        return round(combined_density, 6)
    
    def process_with_real_shark_density(self):
        """Add real shark density to existing attack/non-attack data."""
        
        logger.info("🚀 Starting REAL shark density processing...")
        
        # Load real shark data
        shark_df = self.load_shark_data()
        if shark_df is None:
            logger.error("❌ Failed to load real shark data")
            return
        
        # Process existing data
        input_files = {
            'positive': 'data/processed/positive_with_sst_and_pop.csv',
            'negative': 'data/processed/negative_with_sst_and_pop.csv'
        }
        
        all_data = []
        
        for dataset_type, input_file in input_files.items():
            input_path = Path(input_file)
            
            if not input_path.exists():
                logger.warning(f"❌ File not found: {input_file}")
                continue
                
            logger.info(f"📊 Processing {dataset_type} data with REAL shark densities...")
            
            df = pd.read_csv(input_path)
            logger.info(f"  Loaded {len(df):,} records")
            
            # Calculate real shark density
            shark_densities = []
            
            for idx, row in df.iterrows():
                if (idx + 1) % 1000 == 0:
                    logger.info(f"    Progress: {idx+1:,}/{len(df):,}")
                
                density = self.calculate_real_shark_density(
                    row['Latitude'], 
                    row['Longitude'], 
                    shark_df,
                    radius_km=150  # Slightly larger radius for better coverage
                )
                shark_densities.append(density)
            
            # Add results
            df['Real_Shark_Density'] = shark_densities
            df['Attack_Type'] = dataset_type
            
            # Statistics
            positive_density = sum(1 for d in shark_densities if d > 0)
            avg_density = np.mean([d for d in shark_densities if d > 0]) if positive_density > 0 else 0
            max_density = max(shark_densities) if shark_densities else 0
            
            logger.info(f"  📈 REAL SHARK DENSITY RESULTS:")
            logger.info(f"    Locations with shark data: {positive_density}/{len(shark_densities)} ({positive_density/len(shark_densities)*100:.1f}%)")
            logger.info(f"    Average density: {avg_density:.6f}")
            logger.info(f"    Max density: {max_density:.6f}")
            
            # Save individual file
            output_file = Path(f'data/processed/{dataset_type}_with_real_shark_density.csv')
            df.to_csv(output_file, index=False)
            logger.info(f"✅ Saved: {output_file}")
            
            all_data.append(df)
        
        # Combine and save final dataset
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_file = Path('data/processed/final_real_shark_data.csv')
            combined_df.to_csv(combined_file, index=False)
            
            logger.info("🎉 FINAL REAL SHARK DENSITY SUMMARY:")
            logger.info(f"  Total records: {len(combined_df):,}")
            logger.info(f"  Positive samples: {len(combined_df[combined_df['Attack_Type'] == 'positive']):,}")
            logger.info(f"  Negative samples: {len(combined_df[combined_df['Attack_Type'] == 'negative']):,}")
            
            # Overall real shark density stats
            all_densities = combined_df['Real_Shark_Density']
            with_shark_data = len(all_densities[all_densities > 0])
            
            logger.info(f"  Locations with REAL shark observations: {with_shark_data:,} ({with_shark_data/len(combined_df)*100:.1f}%)")
            if with_shark_data > 0:
                logger.info(f"  Average REAL shark density: {all_densities[all_densities > 0].mean():.6f}")
                logger.info(f"  Max REAL shark density: {all_densities.max():.6f}")
            
            logger.info(f"✅ Final REAL shark data saved: {combined_file}")
            
            # Show sample
            logger.info("\n📋 Sample of REAL shark density data:")
            sample_cols = ['Year', 'Month', 'Latitude', 'Longitude', 'SST_Celsius', 'Population', 'Real_Shark_Density', 'Attack_Type']
            print(combined_df[sample_cols].head(10))

if __name__ == "__main__":
    fetcher = RealSharkDensityFetcher()
    fetcher.process_with_real_shark_density()