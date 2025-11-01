import pandas as pd
from geopy.distance import geodesic

def find_closest_country(lat, lon, country_data):
    """Find the closest country to the given latitude and longitude."""
    closest_country = None
    closest_distance = float('inf')

    for _, row in country_data.iterrows():
        country_coords = (row['Latitude'], row['Longitude'])
        distance = geodesic((lat, lon), country_coords).kilometers
        if distance < closest_distance:
            closest_distance = distance
            closest_country = row['Country']

    return closest_country

def add_population_column(input_file, output_file, country_data):
    """Add population data to the input file based on the closest country."""
    # Load the input file
    data = pd.read_csv(input_file)

    # Add a new column for population
    populations = []
    for _, row in data.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        closest_country = find_closest_country(lat, lon, country_data)
        population = country_data.loc[country_data['Country'] == closest_country, 'Population'].values[0]
        populations.append(population)

    data['Population'] = populations

    # Save the updated file
    data.to_csv(output_file, index=False)
    print(f"Saved updated file with population data: {output_file}")

def main():
    # File paths
    positive_file = "data/processed/positive_with_sst.csv"
    negative_file = "data/processed/negative_with_sst.csv"
    country_file = "data/population/country-capital-lat-long-population.csv"

    positive_output = "data/processed/positive_with_sst_and_pop.csv"
    negative_output = "data/processed/negative_with_sst_and_pop.csv"

    # Load country data
    country_data = pd.read_csv(country_file)

    # Process positive and negative files
    add_population_column(positive_file, positive_output, country_data)
    add_population_column(negative_file, negative_output, country_data)

if __name__ == "__main__":
    main()