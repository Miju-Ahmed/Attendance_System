#!/bin/bash

# Simple MySQL Password Configuration Script

echo "=========================================="
echo "MySQL Password Configuration"
echo "=========================================="
echo ""
echo "This script will help you configure the MySQL password for the backend."
echo ""

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROPERTIES_FILE="$SCRIPT_DIR/backend/src/main/resources/application.properties"

echo "What is your MySQL root password?"
echo "(Press Enter if you don't have a password)"
read -s MYSQL_PASSWORD

# Update the properties file
if [ -f "$PROPERTIES_FILE" ]; then
    # Use sed to replace the password line
    sed -i "s/^spring.datasource.password=.*/spring.datasource.password=$MYSQL_PASSWORD/" "$PROPERTIES_FILE"
    
    echo ""
    echo "✓ Configuration updated!"
    echo ""
    echo "Testing MySQL connection..."
    
    # Test connection
    if mysql -u root -p"$MYSQL_PASSWORD" -e "CREATE DATABASE IF NOT EXISTS attendance_db;" 2>/dev/null; then
        echo "✓ MySQL connection successful!"
        echo "✓ Database 'attendance_db' is ready!"
        echo ""
        echo "You can now start the backend:"
        echo "  cd backend"
        echo "  mvn spring-boot:run"
    else
        echo "⚠️  Could not connect to MySQL or create database."
        echo "Please check your password and make sure MySQL is running:"
        echo "  sudo systemctl start mysql"
    fi
else
    echo "❌ Could not find application.properties file"
    exit 1
fi
