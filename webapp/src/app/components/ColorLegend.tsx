import React from 'react';

export default function ColorLegend() {
  // Generate color legend based on YlOrRd colormap used in Globe component
  const generateLegendColors = () => {
    const colors = [];
    for (let i = 0; i <= 10; i++) {
      const value = i / 10;
      const color = getYlOrRdColor(value);
      colors.push({
        value: value * 100, // Convert to percentage
        color: color
      });
    }
    return colors;
  };

  // YlOrRd colormap function matching the one in Globe.tsx
  const getYlOrRdColor = (value: number): string => {
    const clampedValue = Math.max(0, Math.min(1, value));
    
    if (clampedValue <= 0.125) {
      const t = clampedValue / 0.125;
      return interpolateColor([255, 255, 229], [255, 255, 204], t);
    } else if (clampedValue <= 0.25) {
      const t = (clampedValue - 0.125) / 0.125;
      return interpolateColor([255, 255, 204], [255, 237, 160], t);
    } else if (clampedValue <= 0.375) {
      const t = (clampedValue - 0.25) / 0.125;
      return interpolateColor([255, 237, 160], [254, 217, 118], t);
    } else if (clampedValue <= 0.5) {
      const t = (clampedValue - 0.375) / 0.125;
      return interpolateColor([254, 217, 118], [254, 178, 76], t);
    } else if (clampedValue <= 0.625) {
      const t = (clampedValue - 0.5) / 0.125;
      return interpolateColor([254, 178, 76], [253, 141, 60], t);
    } else if (clampedValue <= 0.75) {
      const t = (clampedValue - 0.625) / 0.125;
      return interpolateColor([253, 141, 60], [252, 78, 42], t);
    } else if (clampedValue <= 0.875) {
      const t = (clampedValue - 0.75) / 0.125;
      return interpolateColor([252, 78, 42], [227, 26, 28], t);
    } else {
      const t = (clampedValue - 0.875) / 0.125;
      return interpolateColor([227, 26, 28], [177, 0, 38], t);
    }
  };

  const interpolateColor = (color1: number[], color2: number[], t: number): string => {
    const r = Math.round(color1[0] + (color2[0] - color1[0]) * t);
    const g = Math.round(color1[1] + (color2[1] - color1[1]) * t);
    const b = Math.round(color1[2] + (color2[2] - color1[2]) * t);
    return `rgb(${r}, ${g}, ${b})`;
  };

  const legendColors = generateLegendColors();

  return (
    <div>
      {/* Gradient bar */}
      <div className="mb-2">
        <div 
          className="w-full h-4 rounded border border-gray-600"
          style={{
            background: `linear-gradient(to right, ${legendColors.map(item => item.color).join(', ')})`
          }}
        />
      </div>

      {/* Labels */}
      <div className="flex justify-between items-center mb-2">
        <span className="text-gray-400 text-xs">0%</span>
        <span className="text-white text-xs font-medium">Risk Level</span>
        <span className="text-gray-400 text-xs">100%</span>
      </div>

      {/* Risk categories */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="text-center">
          <div className="w-full h-1 rounded mb-1" style={{ backgroundColor: legendColors[2].color }} />
          <span className="text-gray-400">Low</span>
        </div>
        <div className="text-center">
          <div className="w-full h-1 rounded mb-1" style={{ backgroundColor: legendColors[5].color }} />
          <span className="text-gray-400">Moderate</span>
        </div>
        <div className="text-center">
          <div className="w-full h-1 rounded mb-1" style={{ backgroundColor: legendColors[8].color }} />
          <span className="text-gray-400">High</span>
        </div>
      </div>
    </div>
  );
}