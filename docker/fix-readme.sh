#!/bin/bash
# This script fixes access to README.md in NVIDIA LaunchPad environments
# It can be run manually in the container with: bash /app/docker/fix-readme.sh

set -e

echo "=== MCTX README Fix ==="
echo "Creating accessible documentation..."

# Create directory for documentation in standard locations
mkdir -p /usr/share/nginx/html
mkdir -p /var/www/html
mkdir -p /home/ubuntu
mkdir -p /root

# Function to copy README files to a directory
copy_readme_files() {
    local DIR=$1
    echo "Copying README files to $DIR..."
    
    # Copy main README files
    cp -f /app/README.md "$DIR/README.md" 2>/dev/null || echo "Could not copy to $DIR/README.md"
    cp -f /app/NVIDIA_LAUNCHPAD.md "$DIR/NVIDIA_LAUNCHPAD.md" 2>/dev/null || echo "Could not copy to $DIR/NVIDIA_LAUNCHPAD.md"
    
    # Create an index.html that redirects to README.md
    cat > "$DIR/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>MCTX Documentation</title>
    <meta http-equiv="refresh" content="0; url=README.md">
</head>
<body>
    <p>Redirecting to <a href="README.md">README.md</a>...</p>
</body>
</html>
EOF
    
    echo "Created files in $DIR"
    ls -la "$DIR"
}

# Copy files to standard web server locations
copy_readme_files "/usr/share/nginx/html"
copy_readme_files "/var/www/html"
copy_readme_files "/home/ubuntu"
copy_readme_files "/root"

# Also make a copy in NVIDIA LaunchPad workspace directory if it exists
if [ -d "/workspace" ]; then
    copy_readme_files "/workspace"
fi

# Try to detect any web server root directories
for DIR in $(find / -type d -name "html" 2>/dev/null | grep -v "node_modules"); do
    copy_readme_files "$DIR"
done

echo "=== Setup Complete ==="
echo "README.md should now be accessible in various locations"
echo "You can also view it directly at: cat /app/README.md"