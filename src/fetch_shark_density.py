#!/usr/bin/env python3
"""
Real Shark Data Downloader
==========================

Downloads real shark observation data from scientific databases:
1. OBIS bulk download (Parquet format)
2. GBIF bulk occurrence download
3. Direct CSV exports from marine databases

NO synthetic data, NO hotspot modeling - only real observed shark data.

Author: GitHub Copilot
Date: November 2025
"""

import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import zipfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealSharkDataFetcher:
    """Downloads real shark observation data from scientific databases."""
    
    def __init__(self):
        self.data_dir = Path('data/shark_observations')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.shark_data = None  # Will hold downloaded real data
        
    def download_obis_shark_data(self) -> Optional[Path]:
        """
        Download real shark observation data from OBIS.
        """
        try:
            logger.info("🦈 Downloading OBIS shark observation data...")
            
            # OBIS shark data export URL (Selachimorpha = sharks)
            url = "https://api.obis.org/occurrence/export?taxonid=11081&format=csv"
            
            output_file = self.data_dir / 'obis_sharks.csv'
            
            logger.info(f"📥 Downloading from OBIS API...")
            response = requests.get(url, timeout=300, stream=True)
            
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Validate file
                try:
                    test_df = pd.read_csv(output_file, nrows=10)
                    if len(test_df) > 0:
                        logger.info(f"✅ Downloaded OBIS shark data: {output_file}")
                        return output_file
                except Exception as e:
                    logger.error(f"❌ Invalid CSV file: {e}")
                    
            else:
                logger.error(f"❌ OBIS download failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ OBIS download error: {e}")
            
        return None
    
    def download_gbif_shark_data(self) -> Optional[Path]:
        """
        Download real shark occurrence data from GBIF.
        """
        try:
            logger.info("🌐 Attempting GBIF shark data download...")
            
            # GBIF simple download for sharks
            # Note: GBIF may require authentication for large downloads
            url = "https://api.gbif.org/v1/occurrence/search?taxon_key=121&limit=50000&format=csv"
            
            output_file = self.data_dir / 'gbif_sharks.csv'
            
            response = requests.get(url, timeout=180)
            
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                # Validate
                try:
                    test_df = pd.read_csv(output_file, nrows=10)
                    if len(test_df) > 0:
                        logger.info(f"✅ Downloaded GBIF shark data: {output_file}")
                        return output_file
                except:
                    pass
                    
            logger.warning("❌ GBIF download failed or requires authentication")
            
        except Exception as e:
            logger.error(f"❌ GBIF download error: {e}")
            
        return None
    
    def download_backup_shark_data(self) -> Optional[Path]:
        """
        Download backup shark data from alternative sources.
        """
        backup_sources = [
            {
                'name': 'Shark Research Committee Data',
                'url': 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-07-17/week16_sharks.csv',
                'file': 'backup_sharks_1.csv'
            },
            {
                'name': 'Marine Life Data',
                'url': 'https://zenodo.org/record/3766915/files/sharks_observations.csv',
                'file': 'backup_sharks_2.csv'
            }
        ]
        
        for source in backup_sources:
            try:
                logger.info(f"📥 Trying backup source: {source['name']}")
                
                response = requests.get(source['url'], timeout=60)
                
                if response.status_code == 200:
                    output_file = self.data_dir / source['file']
                    
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    # Validate
                    try:
                        test_df = pd.read_csv(output_file, nrows=5)
                        if len(test_df) > 0:
                            logger.info(f"✅ Downloaded backup data: {output_file}")
                            return output_file
                    except:
                        continue
                        
            except Exception as e:
                logger.warning(f"❌ Backup source {source['name']} failed: {e}")
                continue
        
        return None
    
    def load_real_shark_data(self) -> Optional[pd.DataFrame]:
        """
        Load real shark observation data from downloaded files.
        """
        # Try to find any downloaded shark data files
        data_files = list(self.data_dir.glob('*.csv'))
        
        if not data_files:
            logger.info("� No existing data files found, attempting downloads...")
            
            # Try downloading from different sources
            downloaded_file = None
            
            # Try OBIS first
            downloaded_file = self.download_obis_shark_data()
            
            # If OBIS fails, try GBIF
            if not downloaded_file:
                downloaded_file = self.download_gbif_shark_data()
            
            # If both fail, try backup sources
            if not downloaded_file:
                downloaded_file = self.download_backup_shark_data()
            
            if not downloaded_file:
                logger.error("❌ Failed to download any real shark data")
                return None
            
            data_files = [downloaded_file]
        
        # Load and combine all available data files
        all_data = []
        
        for data_file in data_files:
            try:
                logger.info(f"📖 Loading data from: {data_file}")
                
                df = pd.read_csv(data_file)
                
                # Standardize column names (different sources use different names)
                column_mapping = {
                    'decimalLatitude': 'latitude',
                    'decimalLongitude': 'longitude',
                    'lat': 'latitude',
                    'lon': 'longitude',
                    'lng': 'longitude',
                    'scientificName': 'species',
                    'species': 'species',
                    'speciesName': 'species',
                    'year': 'year',
                    'eventDate': 'date',
                    'date': 'date'
                }
                
                # Rename columns to standard names
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df = df.rename(columns={old_name: new_name})
                
                # Filter for required columns
                required_cols = ['latitude', 'longitude']
                available_cols = [col for col in required_cols if col in df.columns]
                
                if len(available_cols) >= 2:
                    # Keep only necessary columns
                    keep_cols = available_cols + [col for col in ['species', 'year', 'date'] if col in df.columns]
                    df = df[keep_cols]
                    
                    # Remove invalid coordinates
                    df = df.dropna(subset=['latitude', 'longitude'])
                    df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
                    
                    if len(df) > 0:
                        df['data_source'] = data_file.stem
                        all_data.append(df)
                        logger.info(f"  ✅ Loaded {len(df):,} valid records")
                    else:
                        logger.warning(f"  ❌ No valid coordinates in {data_file}")
                else:
                    logger.warning(f"  ❌ Missing required columns in {data_file}")
                    
            except Exception as e:
                logger.error(f"❌ Error loading {data_file}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"✅ Combined real shark data: {len(combined_df):,} total observations")
            
            # Save combined data for future use
            combined_file = self.data_dir / 'combined_real_shark_data.csv'
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"� Saved combined data: {combined_file}")
            
            return combined_df
        else:
            logger.error("❌ No valid shark observation data found")
            return None
    
    def get_shark_density_for_location(self, lat: float, lon: float, radius_km: float = 100) -> Dict:
        """
        Get shark density for a location using real observation data.
        
        Args:
            lat: Latitude
            lon: Longitude  
            radius_km: Search radius in kilometers
            
        Returns:
            Dictionary with real shark density metrics
        """
        if self.shark_data is None:
            self.shark_data = self.load_real_shark_data()
            
        if self.shark_data is None or len(self.shark_data) == 0:
            return {
                'source': 'none',
                'density': 0,
                'species_count': 0,
                'observations': 0,
                'note': 'No real shark observation data available'
            }
        
        # Calculate distance to all observations
        data = self.shark_data.copy()
        
        # Simple distance calculation (rough approximation)
        lat_diff = data['latitude'] - lat
        lon_diff = data['longitude'] - lon
        distance_deg = np.sqrt(lat_diff**2 + lon_diff**2)
        distance_km = distance_deg * 111  # Rough conversion to km
        
        # Filter observations within radius
        nearby_observations = data[distance_km <= radius_km]
        
        if len(nearby_observations) == 0:
            return {
                'source': 'real_data',
                'density': 0,
                'species_count': 0,
                'observations': 0,
                'search_radius_km': radius_km,
                'note': f'No observations within {radius_km}km radius'
            }
        
        # Calculate metrics
        num_observations = len(nearby_observations)
        area_km2 = np.pi * (radius_km ** 2)
        density = (num_observations / area_km2) * 1000  # Observations per 1000 km²
        
        # Count unique species if species data available
        species_count = 0
        if 'species' in nearby_observations.columns:
            species_count = nearby_observations['species'].nunique()
        
        return {
            'source': 'real_observations',
            'density': round(density, 4),
            'species_count': species_count,
            'observations': num_observations,
            'search_radius_km': radius_km,
            'area_km2': round(area_km2, 1),
            'note': f'Based on {num_observations} real observations within {radius_km}km'
        }

def process_shark_attack_data_with_density():
    """
    Process both positive and negative shark attack data and add shark density information.
    Uses pre-downloaded datasets instead of making thousands of API calls.
    """
    logger.info("🚀 Starting shark density data collection using downloadable datasets...")
    
    # Initialize fetcher
    fetcher = RealSharkDataFetcher()
    
    # Define input and output paths
    input_files = {
        'positive': Path('data/processed/positive_with_sst.csv'),
        'negative': Path('data/processed/negative_with_sst.csv')
    }
    
    output_files = {
        'positive': Path('data/processed/positive_with_shark_density.csv'),
        'negative': Path('data/processed/negative_with_shark_density.csv'),
        'combined': Path('data/processed/combined_shark_data.csv')
    }
    
    # Create output directory
    output_files['positive'].parent.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    for dataset_type, input_file in input_files.items():
        logger.info(f"📊 Processing {dataset_type} dataset: {input_file}")
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            continue
        
        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df):,} records")
        
        # Add shark density columns
        shark_densities = []
        species_counts = []
        observations = []
        data_sources = []
        grid_distances = []
        
        # Process all records (fast grid lookup, no API calls)
        logger.info("  🔍 Looking up shark densities from grid...")
        
        for idx, row in df.iterrows():
            if (idx + 1) % 1000 == 0:
                logger.info(f"    Progress: {idx+1:,}/{len(df):,} records")
            
            lat = row['Latitude']
            lon = row['Longitude']
            
            # Fast grid lookup (no API calls!)
            density_data = fetcher.get_shark_density_for_location(lat, lon)
            
            shark_densities.append(density_data.get('density', 0))
            species_counts.append(density_data.get('species_count', 0))
            observations.append(density_data.get('observations', 0))
            data_sources.append(density_data.get('source', 'none'))
            grid_distances.append(density_data.get('grid_distance_km', 0))
        
        # Add new columns to dataframe
        df['Shark_Density'] = shark_densities
        df['Shark_Species_Count'] = species_counts
        df['Shark_Observations'] = observations
        df['Density_Data_Source'] = data_sources
        df['Grid_Distance_km'] = grid_distances
        df['Attack_Type'] = dataset_type  # positive or negative
        
        # Save individual dataset with density
        df.to_csv(output_files[dataset_type], index=False)
        logger.info(f"✅ Saved {dataset_type} data with shark density: {output_files[dataset_type]}")
        
        # Add to combined dataset
        all_data.append(df)
        
        # Print summary stats
        data_records = sum(1 for d in shark_densities if d > 0)
        avg_density = np.mean([d for d in shark_densities if d > 0]) if data_records > 0 else 0
        max_density = np.max(shark_densities) if shark_densities else 0
        
        logger.info(f"  📈 {dataset_type.title()} density stats:")
        logger.info(f"    Records with shark data: {data_records}/{len(shark_densities)} ({data_records/len(shark_densities)*100:.1f}%)")
        logger.info(f"    Average density: {avg_density:.3f}")
        logger.info(f"    Maximum density: {max_density:.3f}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_files['combined'], index=False)
        logger.info(f"✅ Saved combined dataset: {output_files['combined']}")
        
        # Final summary
        data_records = len(combined_df[combined_df['Shark_Density'] > 0])
        total_records = len(combined_df)
        
        logger.info("🎉 SUMMARY:")
        logger.info(f"  Total records processed: {total_records:,}")
        logger.info(f"  Records with shark data: {data_records:,} ({data_records/total_records*100:.1f}%)")
        logger.info(f"  Positive samples: {len(combined_df[combined_df['Attack_Type'] == 'positive']):,}")
        logger.info(f"  Negative samples: {len(combined_df[combined_df['Attack_Type'] == 'negative']):,}")
        
        densities = combined_df[combined_df['Shark_Density'] > 0]['Shark_Density']
        if len(densities) > 0:
            logger.info(f"  Average shark density: {densities.mean():.3f}")
            logger.info(f"  Max shark density: {densities.max():.3f}")
        
        # Data source breakdown
        source_counts = combined_df['Density_Data_Source'].value_counts()
        logger.info("  Data sources used:")
        for source, count in source_counts.items():
            logger.info(f"    {source}: {count:,} records ({count/len(combined_df)*100:.1f}%)")
    
    return output_files

if __name__ == "__main__":
    # Test the new downloadable approach
    logger.info("🧪 Testing downloadable shark density fetcher...")
    
    fetcher = RealSharkDataFetcher()
    
    # Test locations
    test_locations = [
        (25.7617, -80.1918, "Miami, Florida"),
        (21.3099, -157.8581, "Hawaii"),
        (-34.3553, 18.4697, "Cape Town, South Africa"),
        (-33.8688, 151.2093, "Sydney, Australia"),
        (0, 0, "Equator/Prime Meridian")
    ]
    
    print("\n🦈 DOWNLOADABLE SHARK DENSITY TEST RESULTS:")
    print("=" * 70)
    
    for lat, lon, name in test_locations:
        result = fetcher.get_shark_density_for_location(lat, lon)
        
        print(f"📍 {name}")
        print(f"   Coordinates: {lat:.3f}, {lon:.3f}")
        print(f"   Density: {result.get('density', 0):.3f} sharks/100km²")
        print(f"   Species estimate: {result.get('species_count', 0)}")
        print(f"   Data source: {result.get('source', 'unknown')}")
        if result.get('grid_distance_km'):
            print(f"   Grid distance: {result.get('grid_distance_km', 0):.1f} km")
        if result.get('note'):
            print(f"   Note: {result['note']}")
        print()
    
    print("⚡ This approach is MUCH faster - uses pre-compiled data grid instead of API calls!")
    print("📊 Based on literature-documented shark hotspots and distribution patterns.")
    print()
    
    response = input("🚀 Process full dataset with downloadable shark density? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        logger.info("🚀 Starting full dataset processing...")
        process_shark_attack_data_with_density()
    else:
        print("✅ Test completed. Ready to process when you are!")
    
    def fetch_gbif_shark_data(self, lat: float, lon: float, radius_km: float = 50) -> Dict:
        """
        Fetch shark occurrence data from GBIF.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            
        Returns:
            Dictionary with shark density metrics
        """
        try:
            cache_key = f"gbif_{lat}_{lon}_{radius_km}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # GBIF API parameters
            params = {
                'decimalLatitude': f'{lat-radius_km/111},{lat+radius_km/111}',
                'decimalLongitude': f'{lon-radius_km/111},{lon+radius_km/111}',
                'hasCoordinate': True,
                'hasGeospatialIssue': False,
                'limit': 300
            }
            
            logger.info(f"🌐 Fetching GBIF data for {lat:.3f}, {lon:.3f}")
            
            total_observations = 0
            species_found = set()
            
            # Query each shark species
            for species in self.shark_species[:5]:  # Limit to avoid timeout
                species_params = params.copy()
                species_params['scientificName'] = species
                
                response = self.session.get(
                    'https://api.gbif.org/v1/occurrence/search',
                    params=species_params,
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    count = data.get('count', 0)
                    total_observations += count
                    if count > 0:
                        species_found.add(species)
                
                time.sleep(0.2)  # Rate limiting
            
            density = min(total_observations / (radius_km ** 2), 100)  # Cap at 100
            
            result = {
                'source': 'gbif',
                'density': density,
                'species_count': len(species_found),
                'observations': total_observations
            }
            
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"GBIF fetch error: {e}")
            return {'source': 'gbif', 'density': 0, 'species_count': 0, 'observations': 0}
    
    def fetch_inat_shark_data(self, lat: float, lon: float, radius_km: float = 50) -> Dict:
        """
        Fetch shark observation data from iNaturalist.
        
        Args:
            lat: Latitude
            lon: Longitude  
            radius_km: Search radius in kilometers
            
        Returns:
            Dictionary with shark density metrics from real observations
        """
        try:
            cache_key = f"inat_{lat}_{lon}_{radius_km}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            logger.info(f"🐠 Fetching iNaturalist data for {lat:.3f}, {lon:.3f}")
            
            # iNaturalist API parameters
            params = {
                'lat': lat,
                'lng': lon,
                'radius': radius_km,
                'taxon_name': 'Selachimorpha',  # Sharks
                'verifiable': 'true',
                'per_page': 200
            }
            
            response = self.session.get(
                'https://api.inaturalist.org/v1/observations',
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                result = self._process_inat_data(data, lat, lon, radius_km)
                self.cache[cache_key] = result
                return result
            else:
                logger.warning(f"iNaturalist API failed with status {response.status_code}")
                return {'source': 'inat', 'density': 0, 'species_count': 0, 'observations': 0}
                
        except Exception as e:
            logger.error(f"iNaturalist fetch error: {e}")
            return {'source': 'inat', 'density': 0, 'species_count': 0, 'observations': 0}
    
    def _process_inat_data(self, data: Dict, lat: float, lon: float, radius_km: float) -> Dict:
        """Process iNaturalist API response data."""
        try:
            results = data.get('results', [])
            
            if not results:
                return {'source': 'inat', 'density': 0, 'species_count': 0, 'observations': 0}
            
            # Count valid observations and species
            species_found = set()
            valid_observations = 0
            
            for observation in results:
                if observation.get('location'):
                    valid_observations += 1
                    
                # Extract species information
                taxon = observation.get('taxon', {})
                if taxon.get('name'):
                    species_found.add(taxon['name'])
            
            # Calculate density (observations per 1000 km²)
            area_km2 = np.pi * (radius_km ** 2)
            density = (valid_observations / area_km2) * 1000 if area_km2 > 0 else 0
            
            return {
                'source': 'inat',
                'density': round(density, 3),
                'species_count': len(species_found),
                'observations': valid_observations
            }
            
        except Exception as e:
            logger.error(f"iNaturalist data processing error: {e}")
            return {'source': 'inat', 'density': 0, 'species_count': 0, 'observations': 0}
    
    def _process_obis_data(self, data: Dict, lat: float, lon: float) -> Dict:
        """Process OBIS API response data."""
        try:
            results = data.get('results', [])
            
            if not results:
                return {'source': 'obis', 'density': 0, 'species_count': 0, 'observations': 0}
            
            # Count observations and species
            species_found = set()
            valid_observations = 0
            
            for record in results:
                if record.get('decimalLatitude') and record.get('decimalLongitude'):
                    valid_observations += 1
                    species_name = record.get('scientificName', '').split()[0:2]
                    if len(species_name) == 2:
                        species_found.add(' '.join(species_name))
            
            # Calculate density (observations per 100 km²)
            area_km2 = (50 * 2) ** 2  # Search area
            density = (valid_observations / area_km2) * 100
            
            return {
                'source': 'obis',
                'density': round(density, 2),
                'species_count': len(species_found),
                'observations': valid_observations
            }
            
        except Exception as e:
            logger.error(f"OBIS data processing error: {e}")
            return {'source': 'obis', 'density': 0, 'species_count': 0, 'observations': 0}
    
    def fetch_fishbase_data(self, lat: float, lon: float, radius_km: float = 100) -> Dict:
        """
        Query FishBase API for shark species occurrence data.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            
        Returns:
            Dictionary with shark occurrence data
        """
        try:
            cache_key = f"fishbase_{lat}_{lon}_{radius_km}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            logger.info(f"🐟 Fetching FishBase data for {lat:.3f}, {lon:.3f}")
            
            # FishBase doesn't have a direct coordinate API, so we'll use ecosystem data
            # This is a simplified approach - real implementation would be more complex
            
            # For now, return no data since FishBase API is complex
            return {'source': 'fishbase', 'density': 0, 'species_count': 0, 'observations': 0}
            
        except Exception as e:
            logger.error(f"FishBase fetch error: {e}")
            return {'source': 'fishbase', 'density': 0, 'species_count': 0, 'observations': 0}
    
    def fetch_shark_density(self, lat: float, lon: float, sst: float = None) -> Dict:
        """
        Fetch REAL shark density from scientific databases only.
        
        Args:
            lat: Latitude
            lon: Longitude
            sst: Sea surface temperature (not used, kept for compatibility)
            
        Returns:
            Dictionary with real shark density metrics from scientific sources
        """
        logger.info(f"🦈 Fetching REAL shark density for {lat:.3f}, {lon:.3f}")
        
        results = {}
        
        # Try OBIS first (Ocean Biodiversity Information System - most reliable)
        try:
            obis_data = self.fetch_obis_shark_data(lat, lon)
            results['obis'] = obis_data
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"OBIS failed: {e}")
            results['obis'] = {'source': 'obis', 'density': 0, 'species_count': 0, 'observations': 0}
        
        # Try GBIF (Global Biodiversity Information Facility)
        try:
            gbif_data = self.fetch_gbif_shark_data(lat, lon)
            results['gbif'] = gbif_data
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"GBIF failed: {e}")
            results['gbif'] = {'source': 'gbif', 'density': 0, 'species_count': 0, 'observations': 0}
        
        # Try iNaturalist (citizen science observations)
        try:
            inat_data = self.fetch_inat_shark_data(lat, lon)
            results['inat'] = inat_data
            time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.error(f"iNaturalist failed: {e}")
            results['inat'] = {'source': 'inat', 'density': 0, 'species_count': 0, 'observations': 0}
        
        # Choose best real result (NO SYNTHETIC DATA)
        best_result = self._select_best_real_result(results)
        
        return best_result
    
    def _select_best_real_result(self, results: Dict) -> Dict:
        """Select the best result from real data sources only (no synthetic data)."""
        # Priority: OBIS > GBIF > iNaturalist
        # Only return results with actual observations
        
        for source in ['obis', 'gbif', 'inat']:
            if source in results:
                result = results[source]
                if result.get('observations', 0) > 0:
                    logger.info(f"✅ Using real {source.upper()} data: {result['observations']} observations")
                    return result
        
        # If no real data found, return empty result (NOT synthetic)
        logger.warning("❌ No real shark observation data found for this location")
        return {
            'source': 'none', 
            'density': 0, 
            'species_count': 0, 
            'observations': 0,
            'note': 'No real observations found in scientific databases'
        }

def process_shark_attack_data_with_density():
    """
    Process both positive and negative shark attack data and add shark density information.
    """
    logger.info("🚀 Starting shark density data collection...")
    
    # Initialize fetcher
    fetcher = SharkDensityFetcher()
    
    # Define input and output paths
    input_files = {
        'positive': Path('data/processed/positive_with_sst.csv'),
        'negative': Path('data/processed/negative_with_sst.csv')
    }
    
    output_files = {
        'positive': Path('data/processed/positive_with_shark_density.csv'),
        'negative': Path('data/processed/negative_with_shark_density.csv'),
        'combined': Path('data/processed/combined_shark_data.csv')
    }
    
    # Create output directory
    output_files['positive'].parent.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    for dataset_type, input_file in input_files.items():
        logger.info(f"📊 Processing {dataset_type} dataset: {input_file}")
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            continue
        
        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df):,} records")
        
        # Add shark density columns
        shark_densities = []
        species_counts = []
        observations = []
        data_sources = []
        
        # Process in batches to avoid overwhelming APIs
        batch_size = 50
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"  Batch {batch_num}/{total_batches} - Processing records {i+1}-{min(i+batch_size, len(df))}")
            
            for _, row in batch.iterrows():
                lat = row['Latitude']
                lon = row['Longitude'] 
                sst = row.get('SST_Celsius')
                
                # Fetch shark density
                density_data = fetcher.fetch_shark_density(lat, lon, sst)
                
                shark_densities.append(density_data.get('density', 0))
                species_counts.append(density_data.get('species_count', 0))
                observations.append(density_data.get('observations', 0))
                data_sources.append(density_data.get('source', 'none'))
            
            # Small delay between batches to be respectful to APIs
            if batch_num < total_batches:
                time.sleep(2)
        
        # Add new columns to dataframe
        df['Shark_Density'] = shark_densities
        df['Shark_Species_Count'] = species_counts
        df['Shark_Observations'] = observations
        df['Density_Data_Source'] = data_sources
        df['Attack_Type'] = dataset_type  # positive or negative
        
        # Save individual dataset with density
        df.to_csv(output_files[dataset_type], index=False)
        logger.info(f"✅ Saved {dataset_type} data with shark density: {output_files[dataset_type]}")
        
        # Add to combined dataset
        all_data.append(df)
        
        # Print summary stats
        real_data_count = sum(1 for d in shark_densities if d > 0)
        avg_density = np.mean([d for d in shark_densities if d > 0]) if real_data_count > 0 else 0
        max_density = np.max(shark_densities) if shark_densities else 0
        
        logger.info(f"  📈 {dataset_type.title()} density stats:")
        logger.info(f"    Records with real data: {real_data_count}/{len(shark_densities)} ({real_data_count/len(shark_densities)*100:.1f}%)")
        logger.info(f"    Average density (real data only): {avg_density:.3f}")
        logger.info(f"    Maximum density: {max_density:.3f}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_files['combined'], index=False)
        logger.info(f"✅ Saved combined dataset: {output_files['combined']}")
        
        # Final summary
        real_data_records = len(combined_df[combined_df['Shark_Density'] > 0])
        total_records = len(combined_df)
        
        logger.info("🎉 SUMMARY:")
        logger.info(f"  Total records processed: {total_records:,}")
        logger.info(f"  Records with REAL shark data: {real_data_records:,} ({real_data_records/total_records*100:.1f}%)")
        logger.info(f"  Positive samples: {len(combined_df[combined_df['Attack_Type'] == 'positive']):,}")
        logger.info(f"  Negative samples: {len(combined_df[combined_df['Attack_Type'] == 'negative']):,}")
        
        real_densities = combined_df[combined_df['Shark_Density'] > 0]['Shark_Density']
        if len(real_densities) > 0:
            logger.info(f"  Average shark density (real data): {real_densities.mean():.3f}")
            logger.info(f"  Max shark density: {real_densities.max():.3f}")
        else:
            logger.info("  No real shark density data found")
        
        # Data source breakdown
        source_counts = combined_df['Density_Data_Source'].value_counts()
        logger.info("  Data sources used:")
        for source, count in source_counts.items():
            logger.info(f"    {source}: {count:,} records ({count/len(combined_df)*100:.1f}%)")
    
    return output_files

if __name__ == "__main__":
    # Test with a few coordinates first using REAL data only
    logger.info("🧪 Testing REAL shark density fetcher (no synthetic data)...")
    
    fetcher = SharkDensityFetcher()
    
    test_locations = [
        (25.7617, -80.1918, "Miami, Florida"),
        (21.3099, -157.8581, "Hawaii"),
        (-34.3553, 18.4697, "Cape Town, South Africa"),
        (36.7783, -119.4179, "California Coast"),
        (-33.8688, 151.2093, "Sydney, Australia")
    ]
    
    print("\n🦈 REAL SHARK DENSITY TEST RESULTS:")
    print("=" * 70)
    
    for lat, lon, name in test_locations:
        result = fetcher.fetch_shark_density(lat, lon)
        
        print(f"📍 {name}")
        print(f"   Coordinates: {lat:.3f}, {lon:.3f}")
        print(f"   Density: {result.get('density', 0):.3f}")
        print(f"   Species: {result.get('species_count', 0)}")
        print(f"   Real Observations: {result.get('observations', 0)}")
        print(f"   Data Source: {result.get('source', 'unknown')}")
        if result.get('note'):
            print(f"   Note: {result['note']}")
        print()
    
    # Ask user if they want to process full dataset
    print("⚠️  WARNING: Processing full dataset will make ~15,000 API calls to scientific databases.")
    print("   This may take 2-3 hours and some APIs have daily limits.")
    print("   Only real observation data will be used - no synthetic data.")
    print()
    
    response = input("🚀 Process full dataset with REAL shark density data? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        logger.info("🚀 Starting full dataset processing with REAL data only...")
        process_shark_attack_data_with_density()
    else:
        print("✅ Test completed. Run again when ready to process full dataset!")