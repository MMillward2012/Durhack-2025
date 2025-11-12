'use client';

import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default markers in Next.js
// eslint-disable-next-line @typescript-eslint/no-explicit-any
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface HeatmapData {
  year: number;
  month: number;
  date_string: string;
  lat_range: number[];
  lon_range: number[];
  prob_grid: number[][];
  statistics: {
    mean_probability: number;
    max_probability: number;
    min_probability: number;
    std_probability: number;
    high_risk_count: number;
    medium_risk_count: number;
    low_risk_count: number;
    total_locations: number;
  };
  climate_info: {
    base_sst_mean: number;
    climate_adjustment: number;
    climate_adjusted_sst_mean: number;
    years_since_baseline: number;
  };
  top_locations: Array<{
    lat: number;
    lon: number;
    risk: number;
  }>;
  grid_resolution: {
    lat_points: number;
    lon_points: number;
  };
  generated_at: string;
}

interface SharkAttackMapProps {
  data: HeatmapData | null;
}

// Helper function to get color based on risk probability
const getRiskColor = (probability: number): string => {
  if (probability < 0.005) return '#3B82F6'; // Blue - low risk
  if (probability < 0.01) return '#10B981'; // Green - low-medium risk
  if (probability < 0.02) return '#F59E0B'; // Yellow - medium risk
  if (probability < 0.05) return '#EF4444'; // Red - high risk
  return '#7C2D12'; // Dark red - very high risk
};

// Helper function to get radius based on risk probability
const getRiskRadius = (probability: number): number => {
  const baseRadius = 3;
  const maxRadius = 12;
  const normalizedRisk = Math.min(probability / 0.1, 1); // Normalize to 0-1
  return baseRadius + (normalizedRisk * (maxRadius - baseRadius));
};

// Helper function to convert grid data to point array
const convertGridToPoints = (data: HeatmapData) => {
  const points = [];
  const { lat_range, lon_range, prob_grid } = data;
  
  for (let i = 0; i < lat_range.length; i++) {
    for (let j = 0; j < lon_range.length; j++) {
      if (prob_grid[i] && prob_grid[i][j] !== undefined && prob_grid[i][j] > 0.001) {
        points.push({
          latitude: lat_range[i],
          longitude: lon_range[j],
          risk_probability: prob_grid[i][j]
        });
      }
    }
  }
  
  return points;
};

// Component to animate map updates
function MapUpdater({ data }: { data: HeatmapData | null }) {
  const map = useMap();
  
  useEffect(() => {
    if (data) {
      const points = convertGridToPoints(data);
      if (points.length > 0) {
        // Find the bounds of our data points
        const latitudes = points.map(point => point.latitude);
        const longitudes = points.map(point => point.longitude);
        
        const bounds = L.latLngBounds([
          [Math.min(...latitudes), Math.min(...longitudes)],
          [Math.max(...latitudes), Math.max(...longitudes)]
        ]);
        
        // Fit the map to show all points with some padding
        map.fitBounds(bounds, { padding: [20, 20] });
      }
    }
  }, [data, map]);
  
  return null;
}

export default function SharkAttackMap({ data }: SharkAttackMapProps) {
  const mapRef = useRef<L.Map | null>(null);

  if (!data) {
    return (
      <div className="w-full h-[600px] bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg flex items-center justify-center">
        <div className="text-blue-600">No data available</div>
      </div>
    );
  }

  return (
    <div className="w-full h-[600px] rounded-lg overflow-hidden shadow-xl">
      <MapContainer
        ref={mapRef}
        center={[20, 0]} // Center on the equator initially
        zoom={2}
        style={{ height: '100%', width: '100%' }}
        className="rounded-lg"
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        />
        
        <MapUpdater data={data} />
        
        {/* Render heatmap points */}
        {(() => {
          const points = convertGridToPoints(data);
          return points.slice(0, 1000).map((point, index) => ( // Limit to 1000 points for performance
            <CircleMarker
              key={index}
              center={[point.latitude, point.longitude]}
              radius={getRiskRadius(point.risk_probability)}
              fillColor={getRiskColor(point.risk_probability)}
              color={getRiskColor(point.risk_probability)}
              weight={1}
              opacity={0.8}
              fillOpacity={0.6}
            >
              <Popup>
                <div className="text-sm">
                  <div className="font-semibold text-gray-800 mb-2">
                    🦈 Shark Attack Risk
                  </div>
                  <div className="space-y-1">
                    <div>
                      <strong>Location:</strong> {point.latitude.toFixed(2)}°, {point.longitude.toFixed(2)}°
                    </div>
                    <div>
                      <strong>Risk Probability:</strong> {(point.risk_probability * 100).toFixed(3)}%
                    </div>
                    <div>
                      <strong>Risk Level:</strong>{' '}
                      <span className={`font-semibold ${
                        point.risk_probability < 0.005 ? 'text-blue-600' :
                        point.risk_probability < 0.01 ? 'text-green-600' :
                        point.risk_probability < 0.02 ? 'text-yellow-600' :
                        point.risk_probability < 0.05 ? 'text-red-600' : 'text-red-800'
                      }`}>
                        {point.risk_probability < 0.005 ? 'Low' :
                         point.risk_probability < 0.01 ? 'Low-Medium' :
                         point.risk_probability < 0.02 ? 'Medium' :
                         point.risk_probability < 0.05 ? 'High' : 'Very High'}
                      </span>
                    </div>
                    <div className="text-xs text-gray-600 mt-2">
                      Climate-adjusted prediction for {data.month}/{data.year}
                    </div>
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          ));
        })()}
        
        {/* Highlight top risk locations with special markers */}
        {data.top_locations.slice(0, 5).map((point, index) => (
          <CircleMarker
            key={`top-${index}`}
            center={[point.lat, point.lon]}
            radius={15}
            fillColor="#DC2626"
            color="#FFFFFF"
            weight={3}
            opacity={1}
            fillOpacity={0.8}
          >
            <Popup>
              <div className="text-sm">
                <div className="font-semibold text-red-600 mb-2">
                  ⚠️ High Risk Area #{index + 1}
                </div>
                <div className="space-y-1">
                  <div>
                    <strong>Location:</strong> {point.lat.toFixed(2)}°, {point.lon.toFixed(2)}°
                  </div>
                  <div>
                    <strong>Risk Probability:</strong> {(point.risk * 100).toFixed(3)}%
                  </div>
                  <div className="text-xs text-gray-600 mt-2">
                    Top {index + 1} highest risk location globally
                  </div>
                </div>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
      
      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm rounded-lg p-3 shadow-lg z-[1000]">
        <div className="text-sm font-semibold mb-2">Risk Level</div>
        <div className="space-y-1 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
            <span>Low (&lt;0.5%)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span>Low-Medium (0.5-1%)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span>Medium (1-2%)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span>High (2-5%)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-red-800"></div>
            <span>Very High (&gt;5%)</span>
          </div>
        </div>
      </div>
    </div>
  );
}