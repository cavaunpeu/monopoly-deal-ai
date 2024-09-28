FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY frontend/ .

# Accept build-time environment variable
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Build the application
RUN npm run build

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Default command: start Next.js in production
CMD ["npm", "start", "--", "-H", "0.0.0.0", "-p", "8080"]