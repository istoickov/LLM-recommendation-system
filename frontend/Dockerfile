# Frontend Dockerfile

# Use official Node.js image from Docker Hub
FROM node:16

# Set working directory
WORKDIR /app

# Install dependencies
COPY ./package.json ./package-lock.json app/
RUN npm install

# Copy frontend files into the container
COPY . app/

# Expose frontend port (default React port)
EXPOSE 3000

# Start the React app
CMD ["npm", "start"]
