# Role-Based Employee & Asset Management System

A full-stack web application combining employee/asset management with AI-driven attendance tracking using facial recognition.

## 🚀 Features

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (Admin/User)
- Secure password encryption with BCrypt

### User Features
- Profile management with photo upload
- Asset request and tracking
- Leave application submission
- Personal attendance history
- Application status monitoring

### Admin Features
- User management dashboard
- Asset allocation and management
- Leave approval workflow
- System-wide attendance overview
- AI attendance data synchronization

### AI Attendance System
- Real-time face detection and recognition
- Entry/exit event tracking
- SQLite database for AI data
- Automatic sync to main database

## 🛠️ Technology Stack

### Backend
- **Framework**: Spring Boot 3.x
- **Security**: Spring Security + JWT
- **Database**: MySQL (main), SQLite (AI data)
- **ORM**: Hibernate/JPA
- **Build Tool**: Maven

### Frontend
- **Framework**: React 18 with Vite
- **Routing**: React Router DOM
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **State Management**: React Context API

### AI System
- **Detection**: EfficientDet-D0 (person detection)
- **Face Detection**: SCRFD
- **Recognition**: ArcFace
- **Database**: SQLite

## 📋 Prerequisites

- Java 17 or higher
- Node.js 18 or higher
- MySQL 8.0 or higher
- Python 3.8+ (for AI system)
- Maven 3.6+

## 🔧 Installation & Setup

### 1. Database Setup

```bash
# Create MySQL database
mysql -u root -p
CREATE DATABASE attendance_system;
exit;
```

### 2. Backend Setup

```bash
cd backend

# Update application.properties with your database credentials
# Edit src/main/resources/application.properties

# Build and run
mvn clean install
mvn spring-boot:run
```

The backend will start on `http://localhost:8080`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will start on `http://localhost:5173`

### 4. AI Attendance System

```bash
cd AI_Portion

# Install Python dependencies
pip install -r requirements.txt

# Run the attendance system
python attendance_efficientnetdet.py --source 0
```

## 🔐 Default Credentials

**Admin Account:**
- Email: `miju.ch7@gmail.com`
- Password: `Miju`

## 📁 Project Structure

```
Full_Project/
├── backend/
│   ├── src/main/java/com/attendance/
│   │   ├── config/          # Security & CORS configuration
│   │   ├── controller/      # REST controllers
│   │   ├── dto/             # Data Transfer Objects
│   │   ├── entity/          # JPA entities
│   │   ├── exception/       # Exception handling
│   │   ├── repository/      # Data repositories
│   │   ├── security/        # JWT & authentication
│   │   └── service/         # Business logic
│   └── src/main/resources/
│       ├── application.properties
│       └── data.sql         # Admin seed data
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── admin/       # Admin dashboard components
│       │   ├── auth/        # Login & Register
│       │   ├── layout/      # Sidebar & Layout
│       │   └── user/        # User dashboard components
│       ├── context/         # React Context (Auth)
│       ├── services/        # API services
│       └── utils/           # Axios configuration
└── AI_Portion/
    ├── attendance_efficientnetdet.py
    └── attendance.db        # SQLite database
```

## 🔄 API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `GET /api/auth/me` - Get current user

### Users
- `GET /api/users` - Get all users (Admin)
- `GET /api/users/{id}` - Get user by ID
- `PUT /api/users/{id}` - Update user
- `POST /api/users/{id}/photo` - Upload profile photo

### Assets
- `GET /api/assets` - Get all assets
- `POST /api/assets` - Create asset (Admin)
- `POST /api/assets/{id}/request` - Request asset
- `GET /api/assets/my-assets` - Get user's assets

### Leaves
- `GET /api/leaves` - Get all leaves (Admin)
- `POST /api/leaves` - Apply for leave
- `PUT /api/leaves/{id}/approve` - Approve leave (Admin)
- `PUT /api/leaves/{id}/reject` - Reject leave (Admin)
- `GET /api/leaves/my-leaves` - Get user's leaves

### Attendance
- `GET /api/attendance` - Get all attendance (Admin)
- `GET /api/attendance/my-attendance` - Get user's attendance
- `POST /api/attendance/sync` - Sync from AI system (Admin)

## 🎨 Features Walkthrough

### For Users
1. **Login** - Access the system with credentials
2. **Profile** - Update personal information and upload photo
3. **Assets** - Request available assets
4. **Leave** - Apply for different types of leave
5. **Status** - Track application statuses
6. **Attendance** - View personal attendance records

### For Admins
1. **Dashboard** - Overview of system statistics
2. **Users** - Manage all system users
3. **Leaves** - Approve or reject leave requests
4. **Attendance** - View all attendance and sync AI data
5. **Assets** - Manage and assign assets

## 🔒 Security Features

- JWT token-based authentication
- BCrypt password hashing (strength 12)
- Role-based access control
- Protected API endpoints
- CORS configuration
- Automatic token refresh handling

## 🐛 Troubleshooting

### Backend won't start
- Check MySQL is running
- Verify database credentials in `application.properties`
- Ensure port 8080 is available

### Frontend can't connect to backend
- Verify backend is running on port 8080
- Check CORS configuration in `CorsConfig.java`
- Ensure axios baseURL is correct

### AI system issues
- Verify Python dependencies are installed
- Check SQLite database path in `application.properties`
- Ensure camera/video source is accessible

## 📝 License

This project is for educational purposes.

## 👥 Contributors

- Miju Ahmed

## 📧 Contact

For questions or support, contact: miju.ch7@gmail.com
