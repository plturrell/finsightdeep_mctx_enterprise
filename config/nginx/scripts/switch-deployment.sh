#!/bin/bash
set -e

# MCTX Blue-Green Deployment Switcher
# Switches traffic between blue and green environments

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MCTX Blue-Green Deployment Switcher ===${NC}"

# Get current active deployment
ACTIVE_DEPLOYMENT=$(docker exec nginx-router sh -c 'echo $ACTIVE_DEPLOYMENT')
echo -e "Current active deployment: ${YELLOW}$ACTIVE_DEPLOYMENT${NC}"

# Get target deployment
if [ "$ACTIVE_DEPLOYMENT" == "blue" ]; then
    TARGET_DEPLOYMENT="green"
else
    TARGET_DEPLOYMENT="blue"
fi

# Verify target deployment is healthy
echo -e "${YELLOW}Verifying $TARGET_DEPLOYMENT environment health...${NC}"
HEALTH_CHECK=$(docker exec mctx-$TARGET_DEPLOYMENT curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "Failed")

if [ "$HEALTH_CHECK" != "200" ]; then
    echo -e "${RED}Error: $TARGET_DEPLOYMENT environment is not healthy (Status: $HEALTH_CHECK)${NC}"
    echo "Deployment switch aborted."
    exit 1
fi

echo -e "${GREEN}$TARGET_DEPLOYMENT environment is healthy!${NC}"

# Switch to target deployment
echo -e "${YELLOW}Switching traffic to $TARGET_DEPLOYMENT environment...${NC}"
docker exec -e ACTIVE_DEPLOYMENT=$TARGET_DEPLOYMENT nginx-router sh -c "echo 'export ACTIVE_DEPLOYMENT=$TARGET_DEPLOYMENT' > /etc/nginx/env.sh && nginx -s reload"

# Verify switch was successful
echo -e "${YELLOW}Verifying traffic switch...${NC}"
sleep 3
NEW_ACTIVE_DEPLOYMENT=$(docker exec nginx-router sh -c 'echo $ACTIVE_DEPLOYMENT')

if [ "$NEW_ACTIVE_DEPLOYMENT" == "$TARGET_DEPLOYMENT" ]; then
    echo -e "${GREEN}Success! Traffic is now directed to the $TARGET_DEPLOYMENT environment.${NC}"
    echo -e "API is accessible at: http://localhost/api"
    echo -e "Visualization is accessible at: http://localhost:805$([ "$TARGET_DEPLOYMENT" == "blue" ] && echo "1" || echo "2")"
else
    echo -e "${RED}Error: Traffic switch was unsuccessful.${NC}"
    echo -e "Current active deployment: $NEW_ACTIVE_DEPLOYMENT"
    exit 1
fi

echo -e "${YELLOW}Note: The previous environment ($ACTIVE_DEPLOYMENT) is still running and can be accessed directly.${NC}"
echo -e "To roll back, simply run this script again."