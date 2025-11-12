'use client';

import React from 'react';
import { MapContainer, TileLayer, ImageOverlay, useMapEvents } from 'react-leaflet';

interface LeafletMapProps {
  heatmapUrl: string | null;
  onMapClick: (lat: number, lon: number) => void;
  bounds: [[number, number], [number, number]];
}

function MapClickHandler({ onMapClick }: { onMapClick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click: (e) => {
      onMapClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function LeafletMap({ heatmapUrl, onMapClick, bounds }: LeafletMapProps) {
  return (
    <div className="absolute inset-0 w-full h-full z-0">
      <MapContainer
        center={[0, 0]}
        zoom={2}
        bounds={bounds}
        style={{ height: '100%', width: '100%' }}
        zoomControl={true}
        attributionControl={false}
        maxBounds={[[-85, -180], [85, 180]]}
        maxBoundsViscosity={1.0}
        worldCopyJump={false}
        minZoom={2}
      >
        {/* ESRI World Imagery satellite tiles */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution='Tiles &copy; Esri'
          maxZoom={18}
          noWrap={true}
          bounds={[[-85, -180], [85, 180]]}
        />
        
        {/* Heatmap overlay - bounds exactly match world coordinates */}
        {heatmapUrl && (
          <ImageOverlay
            url={heatmapUrl}
            bounds={bounds}
            opacity={1}
          />
        )}
        
        <MapClickHandler onMapClick={onMapClick} />
      </MapContainer>
    </div>
  );
}
