# Quick Start Guide

## Backend Setup Issue - MySQL Connection

The backend failed to start because it couldn't connect to MySQL. Here's how to fix it:

### Option 1: Start MySQL Service

```bash
# Check if MySQL is running
sudo systemctl status mysql

# If not running, start it
sudo systemctl start mysql
```

### Option 2: Use the Setup Script

```bash
cd /run/media/miju_chowdhury/Miju/WorkSpace/Attendance_System/Full_Project
./setup-database.sh
```

This script will:
- Check if MySQL is running
- Create the `attendance_db` database
- Guide you through configuration

### Option 3: Manual Database Setup

```bash
# Login to MySQL
mysql -u root -p

# Create database
CREATE DATABASE attendance_db;
exit;
```

### Update Configuration

Edit `backend/src/main/resources/application.properties`:

```properties
spring.datasource.password=YOUR_MYSQL_PASSWORD
```

Replace `YOUR_MYSQL_PASSWORD` with your actual MySQL root password.

### Start Backend

```bash
cd backend
mvn spring-boot:run
```

Or with password override:

```bash
mvn spring-boot:run -Dspring-boot.run.arguments=--spring.datasource.password=YOUR_PASSWORD
```

## Current Status

✅ **Frontend**: Running on http://localhost:5173  
⚠️  **Backend**: Needs MySQL configuration  
📦 **Database**: attendance_db (needs to be created)

## Default Credentials

**Admin Login:**
- Email: `miju.ch7@gmail.com`
- Password: `Miju`

(Admin user will be auto-created on first backend startup)
