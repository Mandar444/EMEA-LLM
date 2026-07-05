# Stage 1: Build the React Application
FROM node:22-alpine AS build

WORKDIR /app

# Copy package descriptors
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY frontend/ ./

# Build production SPA assets
RUN npm run build

# Stage 2: Serve static build with Nginx
FROM nginx:alpine

# Copy built assets to default nginx folder
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
