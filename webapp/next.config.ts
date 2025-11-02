import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack: (config) => {
    // Handle Cesium source maps
    config.module = config.module || {};
    config.module.unknownContextCritical = false;
    config.module.unknownContextRegExp = /\/cesium\/cesium\/Source\/Core\/buildModuleUrl\.js/;
    
    return config;
  },
};

export default nextConfig;
