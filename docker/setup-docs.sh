#!/bin/bash
set -e

# Create documentation directory
mkdir -p /docs

# Copy README files to accessible location
cp /app/README.md /docs/README.md
cp /app/NVIDIA_LAUNCHPAD.md /docs/NVIDIA_LAUNCHPAD.md
cp /app/docker_README.md /docs/DOCKER_README.md
cp /app/docker_deployment_guide.md /docs/DOCKER_DEPLOYMENT_GUIDE.md

# Create an HTML index for easy browsing
cat > /docs/index.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>FinsightDeep MCTX Enterprise Documentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 10px;
        }
        a {
            color: #0366d6;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>FinsightDeep MCTX Enterprise Documentation</h1>
    <p>Welcome to the FinsightDeep MCTX Enterprise documentation. This page provides links to key documentation files.</p>
    
    <h2>Documentation</h2>
    <ul>
        <li><a href="README.md">README.md</a> - Main project overview and documentation</li>
        <li><a href="NVIDIA_LAUNCHPAD.md">NVIDIA_LAUNCHPAD.md</a> - Guide for NVIDIA LaunchPad users</li>
        <li><a href="DOCKER_README.md">DOCKER_README.md</a> - Docker deployment information</li>
        <li><a href="DOCKER_DEPLOYMENT_GUIDE.md">DOCKER_DEPLOYMENT_GUIDE.md</a> - Detailed Docker deployment guide</li>
    </ul>
    
    <h2>Services</h2>
    <ul>
        <li><a href="http://localhost:8050" target="_blank">Visualization Dashboard</a> (Port 8050)</li>
        <li><a href="http://localhost:8000" target="_blank">API Server</a> (Port 8000)</li>
        <li><a href="http://localhost:8051" target="_blank">Secondary Visualization</a> (Port 8051)</li>
        <li><a href="http://localhost:9090" target="_blank">Prometheus</a> (Port 9090)</li>
        <li><a href="http://localhost:3001" target="_blank">Grafana</a> (Port 3001)</li>
    </ul>
</body>
</html>
EOF

echo "Documentation setup complete. Access docs at /docs/"