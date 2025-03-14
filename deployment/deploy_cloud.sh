#!/bin/bash

# Stop existing containers
docker-compose down

# Pull latest repository changes
git pull origin main

# Rebuild and restart services
docker-compose up --build -d

# Monitor logs
docker logs -f alpha_generator
