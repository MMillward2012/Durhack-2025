"""
Download Real Global SST Data for Shark Attack Prediction (2018-2024)
"""

from src.fetch_global_sst import GlobalSSTFetcher
from datetime import datetime

print("="*70)
print("DOWNLOADING REAL NOAA SST DATA FOR SHARK ATTACK PROJECT")
print("="*70)

fetcher = GlobalSSTFetcher()

# Download monthly SST data for 2018-2024
# This matches the date range in your shark attack dataset
print("\n📅 Date Range: 2018-01-01 to 2024-06-01")
print("📊 Sampling: Monthly (first day of each month)")
print("🌍 Coverage: Global ocean, 0.5° resolution")
print("💾 Expected size: ~10 million records (~500 MB)")
print("\n⏱️  Estimated time: 15-20 minutes")
print("\nStarting download...")
print("="*70)

try:
    df = fetcher.fetch_date_range(
        start_date='2018-01-01',
        end_date='2024-06-01',  # June 2024 should be available
        sample_frequency='monthly'
    )
    
    if df is not None:
        print("\n" + "="*70)
        print("✓ DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique dates: {df['date'].nunique()}")
        print(f"Lat range: {df['latitude'].min():.1f}° to {df['latitude'].max():.1f}°")
        print(f"Lon range: {df['longitude'].min():.1f}° to {df['longitude'].max():.1f}°")
        print(f"SST range: {df['sst_mean'].min():.1f}°C to {df['sst_mean'].max():.1f}°C")
        print("\n✓ Real NOAA SST data is now ready for use!")
        print("✓ You can now match this with your shark attack data")
    else:
        print("\n✗ Download failed - see errors above")
        
except KeyboardInterrupt:
    print("\n\n⚠️  Download interrupted by user")
    print("Partial data may have been saved in data/raw/")
except Exception as e:
    print(f"\n✗ Error: {e}")
