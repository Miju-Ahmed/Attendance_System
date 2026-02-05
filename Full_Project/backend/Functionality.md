# Backend Functionality Documentation

This document describes all the functions and their functionality in the Attendance System backend.

---

## Table of Contents

1. [Authentication Module](#1-authentication-module)
2. [User Management Module](#2-user-management-module)
3. [Attendance Tracking Module](#3-attendance-tracking-module)
4. [Leave Management Module](#4-leave-management-module)
5. [Asset Management Module](#5-asset-management-module)
6. [Admin Operations Module](#6-admin-operations-module)
7. [Security & JWT Module](#7-security--jwt-module)

---

## 1. Authentication Module

### Controller: `AuthController`
**Base Path:** `/api/auth`

#### 1.1 Login
- **Endpoint:** `POST /api/auth/login`
- **Function:** `login(LoginRequest loginRequest)`
- **Description:** Authenticates a user with email and password credentials
- **Input:** 
  - Email address
  - Password
- **Process:**
  - Validates credentials using Spring Security's AuthenticationManager
  - Generates JWT token upon successful authentication
  - Retrieves user details from database
- **Output:** JWT token, user ID, name, email, and role
- **Access:** Public

#### 1.2 Register
- **Endpoint:** `POST /api/auth/register`
- **Function:** `register(RegisterRequest registerRequest)`
- **Description:** Registers a new user in the system
- **Input:**
  - Name
  - Email
  - Password
  - Phone (optional)
  - Address (optional)
- **Process:**
  - Checks if email already exists
  - Encodes password using BCrypt
  - Assigns default role as USER
  - Auto-generates stable_id for face recognition (max existing stable_id + 1)
  - Saves user to database
- **Output:** Success message
- **Access:** Public

#### 1.3 Get Current User
- **Endpoint:** `GET /api/auth/me`
- **Function:** `getCurrentUser()`
- **Description:** Retrieves the currently authenticated user's information
- **Input:** JWT token (from Authorization header)
- **Process:**
  - Extracts email from SecurityContext
  - Fetches user details from database
- **Output:** Complete user object
- **Access:** Authenticated users only

### Service: `AuthService`

#### 1.4 Login Service
- **Function:** `login(LoginRequest loginRequest)`
- **Description:** Business logic for user authentication
- **Process:**
  - Authenticates using AuthenticationManager
  - Sets authentication in SecurityContext
  - Generates JWT token via JwtTokenProvider
  - Logs login attempts and results
- **Returns:** JwtResponse with token and user details

#### 1.5 Register Service
- **Function:** `register(RegisterRequest registerRequest)`
- **Description:** Business logic for user registration
- **Process:**
  - Validates email uniqueness
  - Creates new User entity
  - Encodes password
  - Assigns stable_id sequentially
  - Logs registration events
- **Returns:** Saved User entity

---

## 2. User Management Module

### Controller: `UserController`
**Base Path:** `/api/users`

#### 2.1 Get All Users
- **Endpoint:** `GET /api/users`
- **Function:** `getAllUsers()`
- **Description:** Retrieves a list of all registered users
- **Access:** ADMIN only
- **Output:** List of all users

#### 2.2 Get User By ID
- **Endpoint:** `GET /api/users/{id}`
- **Function:** `getUserById(Long id)`
- **Description:** Retrieves a specific user's details by their ID
- **Input:** User ID (path parameter)
- **Access:** Authenticated users
- **Output:** User object

#### 2.3 Update User
- **Endpoint:** `PUT /api/users/{id}`
- **Function:** `updateUser(Long id, User userDetails)`
- **Description:** Updates user profile information
- **Input:**
  - User ID (path parameter)
  - Updated user details (name, phone, address)
- **Authorization:**
  - Users can only update their own profile
  - Admins can update any user's profile
- **Output:** Updated user object

#### 2.4 Upload Profile Photo
- **Endpoint:** `POST /api/users/{id}/photo`
- **Function:** `uploadProfilePhoto(Long id, MultipartFile file)`
- **Description:** Uploads a profile photo for a user
- **Input:**
  - User ID (path parameter)
  - Image file (multipart/form-data)
- **Process:**
  - Creates upload directory if not exists
  - Generates unique filename using UUID
  - Saves file to `./uploads/profiles/`
  - Updates user's profilePhoto field
- **Authorization:**
  - Users can only upload their own photo
  - Admins can upload for any user
- **Output:** Success message

#### 2.5 Get Profile Photo
- **Endpoint:** `GET /api/users/photo/{filename}`
- **Function:** `getProfilePhoto(String filename)`
- **Description:** Retrieves a user's profile photo
- **Input:** Filename (path parameter)
- **Output:** Image bytes with JPEG content type
- **Access:** Authenticated users

### Service: `UserService`

#### 2.6 Get All Users Service
- **Function:** `getAllUsers()`
- **Description:** Fetches all users from database
- **Returns:** List of User entities

#### 2.7 Get User By ID Service
- **Function:** `getUserById(Long id)`
- **Description:** Fetches a specific user by ID
- **Throws:** ResourceNotFoundException if user not found
- **Returns:** User entity

#### 2.8 Update User Service
- **Function:** `updateUser(Long id, User userDetails)`
- **Description:** Updates user profile fields (name, phone, address)
- **Process:**
  - Fetches existing user
  - Updates only non-null fields
  - Saves to database
- **Returns:** Updated User entity

#### 2.9 Upload Profile Photo Service
- **Function:** `uploadProfilePhoto(Long userId, MultipartFile file)`
- **Description:** Handles profile photo upload logic
- **Process:**
  - Creates upload directory
  - Generates UUID-based filename
  - Copies file to disk
  - Updates user record
- **Returns:** Updated User entity

#### 2.10 Get Profile Photo Service
- **Function:** `getProfilePhoto(String filename)`
- **Description:** Reads profile photo from disk
- **Throws:** ResourceNotFoundException if file not found
- **Returns:** Byte array of image data

---

## 3. Attendance Tracking Module

### Controller: `AttendanceController`
**Base Path:** `/api/attendance`

#### 3.1 Get All Attendance
- **Endpoint:** `GET /api/attendance`
- **Function:** `getAllAttendance()`
- **Description:** Retrieves all attendance records
- **Access:** ADMIN only
- **Output:** List of all attendance records

#### 3.2 Get Attendance By User ID
- **Endpoint:** `GET /api/attendance/user/{userId}`
- **Function:** `getAttendanceByUserId(Long userId)`
- **Description:** Retrieves attendance records for a specific user
- **Input:** User ID (path parameter)
- **Access:** ADMIN only
- **Output:** List of attendance records for the user

#### 3.3 Get My Attendance
- **Endpoint:** `GET /api/attendance/my-attendance`
- **Function:** `getMyAttendance()`
- **Description:** Retrieves attendance records for the currently logged-in user
- **Access:** Authenticated users
- **Output:** List of user's own attendance records

#### 3.4 Get Attendance By Date Range
- **Endpoint:** `GET /api/attendance/range`
- **Function:** `getAttendanceByRange(LocalDateTime start, LocalDateTime end)`
- **Description:** Retrieves attendance records within a date range
- **Input:**
  - start: Start date-time (query parameter)
  - end: End date-time (query parameter)
- **Access:** ADMIN only
- **Output:** List of attendance records in the range

#### 3.5 Sync From SQLite
- **Endpoint:** `POST /api/attendance/sync`
- **Function:** `syncFromSQLite()`
- **Description:** Synchronizes attendance data from AI script's SQLite database to PostgreSQL
- **Process:**
  - Connects to SQLite database
  - Reads attendance records
  - Matches stable_id to users
  - Checks for duplicates (within 1-second window)
  - Inserts new records
- **Access:** ADMIN only
- **Output:** Count of synced records

### Service: `AttendanceService`

#### 3.6 Get All Attendance Service
- **Function:** `getAllAttendance()`
- **Description:** Fetches all attendance records
- **Returns:** List of Attendance entities

#### 3.7 Get User Attendance Service
- **Function:** `getUserAttendance(User user)`
- **Description:** Fetches attendance records for a specific user, ordered by timestamp descending
- **Returns:** List of Attendance entities

#### 3.8 Get Attendance By Date Range Service
- **Function:** `getAttendanceByDateRange(LocalDateTime start, LocalDateTime end)`
- **Description:** Fetches attendance records between two timestamps
- **Returns:** List of Attendance entities

#### 3.9 Sync From SQLite Service
- **Function:** `syncFromSQLite()`
- **Description:** Synchronizes attendance data from SQLite to PostgreSQL
- **Process:**
  - Establishes JDBC connection to SQLite
  - Queries attendance table
  - Parses timestamps
  - Finds users by stable_id
  - Checks for duplicate records
  - Creates new Attendance entities
  - Logs sync results
- **Returns:** Count of synced records
- **Throws:** RuntimeException on database errors

---

## 4. Leave Management Module

### Controller: `LeaveController`
**Base Path:** `/api/leaves`

#### 4.1 Get All Leaves
- **Endpoint:** `GET /api/leaves`
- **Function:** `getAllLeaves()`
- **Description:** Retrieves all leave applications
- **Access:** ADMIN only
- **Output:** List of all leave records

#### 4.2 Get Pending Leaves
- **Endpoint:** `GET /api/leaves/pending`
- **Function:** `getPendingLeaves()`
- **Description:** Retrieves all pending leave applications
- **Access:** ADMIN only
- **Output:** List of pending leave records

#### 4.3 Apply Leave
- **Endpoint:** `POST /api/leaves`
- **Function:** `applyLeave(LeaveRequest request)`
- **Description:** Submits a new leave application
- **Input:**
  - Leave type
  - Start date
  - End date
  - Reason
- **Process:**
  - Creates new Leave entity
  - Sets status to PENDING
  - Associates with current user
- **Access:** Authenticated users
- **Output:** Success message

#### 4.4 Approve Leave
- **Endpoint:** `PUT /api/leaves/{id}/approve`
- **Function:** `approveLeave(Long id)`
- **Description:** Approves a leave application
- **Input:** Leave ID (path parameter)
- **Process:**
  - Fetches leave by ID
  - Updates status to APPROVED
- **Access:** ADMIN only
- **Output:** Updated leave record

#### 4.5 Reject Leave
- **Endpoint:** `PUT /api/leaves/{id}/reject`
- **Function:** `rejectLeave(Long id)`
- **Description:** Rejects a leave application
- **Input:** Leave ID (path parameter)
- **Process:**
  - Fetches leave by ID
  - Updates status to REJECTED
- **Access:** ADMIN only
- **Output:** Updated leave record

#### 4.6 Get My Leaves
- **Endpoint:** `GET /api/leaves/my-leaves`
- **Function:** `getMyLeaves()`
- **Description:** Retrieves leave applications for the current user
- **Access:** Authenticated users
- **Output:** List of user's leave records

### Service: `LeaveService`

#### 4.7 Get All Leaves Service
- **Function:** `getAllLeaves()`
- **Description:** Fetches all leave records
- **Returns:** List of Leave entities

#### 4.8 Get Leave By ID Service
- **Function:** `getLeaveById(Long id)`
- **Description:** Fetches a specific leave by ID
- **Throws:** ResourceNotFoundException if not found
- **Returns:** Leave entity

#### 4.9 Apply Leave Service
- **Function:** `applyLeave(LeaveRequest request, User user)`
- **Description:** Creates a new leave application
- **Process:**
  - Creates Leave entity
  - Sets all fields from request
  - Sets status to PENDING
  - Associates with user
- **Returns:** Saved Leave entity

#### 4.10 Approve Leave Service
- **Function:** `approveLeave(Long leaveId)`
- **Description:** Updates leave status to APPROVED
- **Returns:** Updated Leave entity

#### 4.11 Reject Leave Service
- **Function:** `rejectLeave(Long leaveId)`
- **Description:** Updates leave status to REJECTED
- **Returns:** Updated Leave entity

#### 4.12 Get User Leaves Service
- **Function:** `getUserLeaves(User user)`
- **Description:** Fetches leaves for a user, ordered by creation date descending
- **Returns:** List of Leave entities

#### 4.13 Get Pending Leaves Service
- **Function:** `getPendingLeaves()`
- **Description:** Fetches all leaves with PENDING status
- **Returns:** List of Leave entities

---

## 5. Asset Management Module

### Controller: `AssetController`
**Base Path:** `/api/assets`

#### 5.1 Get All Assets
- **Endpoint:** `GET /api/assets`
- **Function:** `getAllAssets()`
- **Description:** Retrieves all assets in the system
- **Access:** Authenticated users
- **Output:** List of all assets

#### 5.2 Get Asset By ID
- **Endpoint:** `GET /api/assets/{id}`
- **Function:** `getAssetById(Long id)`
- **Description:** Retrieves a specific asset by ID
- **Input:** Asset ID (path parameter)
- **Access:** Authenticated users
- **Output:** Asset object

#### 5.3 Create Asset
- **Endpoint:** `POST /api/assets`
- **Function:** `createAsset(AssetRequest request)`
- **Description:** Creates a new asset in the system
- **Input:**
  - Asset name
  - Asset type
- **Process:**
  - Creates new Asset entity
  - Sets status to AVAILABLE
- **Access:** ADMIN only
- **Output:** Created asset object

#### 5.4 Request Asset
- **Endpoint:** `POST /api/assets/{id}/request`
- **Function:** `requestAsset(Long id)`
- **Description:** Allows a user to request an asset
- **Input:** Asset ID (path parameter)
- **Process:**
  - Validates asset is AVAILABLE
  - Updates status to REQUESTED
  - Associates with current user
- **Access:** Authenticated users
- **Output:** Success message

#### 5.5 Assign Asset
- **Endpoint:** `PUT /api/assets/{id}/assign`
- **Function:** `assignAsset(Long id, Long userId)`
- **Description:** Assigns an asset to a user
- **Input:**
  - Asset ID (path parameter)
  - User ID (query parameter)
- **Process:**
  - Updates asset status to ASSIGNED
- **Access:** ADMIN only
- **Output:** Updated asset object

#### 5.6 Get My Assets
- **Endpoint:** `GET /api/assets/my-assets`
- **Function:** `getMyAssets()`
- **Description:** Retrieves assets assigned to the current user
- **Access:** Authenticated users
- **Output:** List of user's assets

### Service: `AssetService`

#### 5.7 Get All Assets Service
- **Function:** `getAllAssets()`
- **Description:** Fetches all assets
- **Returns:** List of Asset entities

#### 5.8 Get Asset By ID Service
- **Function:** `getAssetById(Long id)`
- **Description:** Fetches a specific asset by ID
- **Throws:** ResourceNotFoundException if not found
- **Returns:** Asset entity

#### 5.9 Create Asset Service
- **Function:** `createAsset(AssetRequest request)`
- **Description:** Creates a new asset
- **Process:**
  - Creates Asset entity
  - Sets name and type
  - Sets status to AVAILABLE
- **Returns:** Saved Asset entity

#### 5.10 Request Asset Service
- **Function:** `requestAsset(Long assetId, User user)`
- **Description:** Processes asset request
- **Validation:** Throws exception if asset not AVAILABLE
- **Process:**
  - Updates status to REQUESTED
  - Associates with user
- **Returns:** Updated Asset entity

#### 5.11 Assign Asset Service
- **Function:** `assignAsset(Long assetId, Long userId)`
- **Description:** Assigns asset to user
- **Process:**
  - Updates status to ASSIGNED
- **Returns:** Updated Asset entity

#### 5.12 Get User Assets Service
- **Function:** `getUserAssets(User user)`
- **Description:** Fetches assets assigned to a user
- **Returns:** List of Asset entities

---

## 6. Admin Operations Module

### Controller: `AdminController`
**Base Path:** `/api/admin`

#### 6.1 Get All Users (Admin)
- **Endpoint:** `GET /api/admin/users`
- **Function:** `getAllUsers()`
- **Description:** Retrieves all users (admin endpoint)
- **Access:** ADMIN only
- **Output:** List of all users

#### 6.2 Get All Users With Secrets
- **Endpoint:** `GET /api/admin/users-with-secrets`
- **Function:** `getAllUsersWithSecrets()`
- **Description:** Retrieves all users including their visible passwords
- **Warning:** Security-sensitive endpoint for demonstration purposes
- **Access:** ADMIN only
- **Output:** List of users with visiblePassword field

#### 6.3 Sync Users From AI
- **Endpoint:** `POST /api/admin/sync-users`
- **Function:** `syncUsers()`
- **Description:** Synchronizes users from AI script's face_embeddings table
- **Process:**
  - Reads distinct person_name and stable_id from face_embeddings
  - Creates users with auto-generated emails (name@employee.com)
  - Generates passwords (Name + "123!")
  - Stores visible passwords
  - Skips existing users
- **Access:** ADMIN only
- **Output:** Sync statistics (added, skipped, errors, logs)

#### 6.4 Upload Video
- **Endpoint:** `POST /api/admin/upload-video`
- **Function:** `uploadVideo(MultipartFile file)`
- **Description:** Uploads a video file for AI processing
- **Input:** Video file (multipart/form-data)
- **Process:**
  - Saves file to `../AI_Portion/uploads/`
  - Generates unique filename with timestamp
  - Triggers asynchronous video processing
  - Runs in non-interactive mode with output file
- **Access:** ADMIN only
- **Output:** Success message with processed video filename

#### 6.5 Get Attendance Logs
- **Endpoint:** `GET /api/admin/attendance-logs`
- **Function:** `getAttendanceLogs()`
- **Description:** Retrieves attendance logs directly from SQLite database
- **Process:**
  - Queries SQLite attendance table
  - Returns last 100 records
- **Access:** ADMIN only
- **Output:** List of attendance log entries

#### 6.6 Start Live Stream
- **Endpoint:** `POST /api/admin/live-stream`
- **Function:** `startLiveStream()`
- **Description:** Starts live webcam stream processing
- **Process:**
  - Triggers AI script with source "0" (default webcam)
  - Runs in interactive mode
  - No output file generated
- **Access:** ADMIN only
- **Output:** Success message

### Service: `SyncService`

#### 6.7 Sync Users From AI Service
- **Function:** `syncUsersFromAI()`
- **Description:** Synchronizes users from face_embeddings table
- **Process:**
  - Step 1: Reads face_embeddings using raw JDBC
  - Step 2: Creates users using JPA (after closing JDBC connection)
  - Generates email: name.lowercase@employee.com
  - Generates password: Name + "123!"
  - Encodes password with BCrypt
  - Stores visible password
  - Handles errors gracefully (partial success)
- **Returns:** Map with status, counts, and logs

#### 6.8 Get Attendance Logs Service
- **Function:** `getAttendanceLogs()`
- **Description:** Fetches attendance logs from SQLite
- **Process:**
  - Uses raw JDBC to query SQLite
  - Retrieves last 100 records
  - Orders by timestamp descending
- **Returns:** List of attendance log maps

### Service: `VideoProcessingService`

#### 6.9 Process Video Service
- **Function:** `processVideo(String videoFilePath, String outputFilePath, boolean isInteractive)`
- **Description:** Processes video using AI Python script
- **Input:**
  - videoFilePath: Path to video or "0" for webcam
  - outputFilePath: Output file path (null for no output)
  - isInteractive: Whether to show interactive display
- **Process:**
  - Validates file exists (unless webcam)
  - Constructs Python command with arguments
  - Sets working directory to AI_Portion
  - Executes asynchronously
  - Captures output and logs
  - Waits for completion
- **Command Arguments:**
  - `--source`: Video file path or "0"
  - `--device`: cpu
  - `--output`: Output file path (if specified)
  - `--no-interaction`: If not interactive
  - `--skip-display`: If not interactive
- **Returns:** CompletableFuture with output path or error
- **Annotation:** @Async for non-blocking execution

---

## 7. Security & JWT Module

### Component: `JwtTokenProvider`

#### 7.1 Generate Token
- **Function:** `generateToken(Authentication authentication)`
- **Description:** Generates a JWT token for authenticated user
- **Process:**
  - Extracts UserDetails from Authentication
  - Creates JWT with subject (username/email)
  - Sets issued date and expiration date
  - Signs with HMAC secret key
- **Returns:** JWT token string

#### 7.2 Get Username From Token
- **Function:** `getUsernameFromToken(String token)`
- **Description:** Extracts username (email) from JWT token
- **Process:**
  - Parses JWT token
  - Verifies signature
  - Extracts subject claim
- **Returns:** Username/email string

#### 7.3 Validate Token
- **Function:** `validateToken(String token)`
- **Description:** Validates JWT token signature and expiration
- **Process:**
  - Attempts to parse and verify token
  - Catches JWT exceptions
- **Returns:** Boolean (true if valid, false otherwise)

#### 7.4 Get Signing Key
- **Function:** `getSigningKey()`
- **Description:** Generates HMAC signing key from secret
- **Returns:** SecretKey for JWT signing

### Filter: `JwtAuthenticationFilter`

#### 7.5 JWT Authentication Filter
- **Description:** Intercepts HTTP requests to validate JWT tokens
- **Process:**
  - Extracts JWT from Authorization header (Bearer token)
  - Validates token using JwtTokenProvider
  - Extracts username from token
  - Loads UserDetails from database
  - Sets authentication in SecurityContext
  - Continues filter chain

### Service: `UserDetailsServiceImpl`

#### 7.6 Load User By Username
- **Function:** `loadUserByUsername(String email)`
- **Description:** Loads user details for Spring Security authentication
- **Process:**
  - Fetches user by email
  - Creates UserDetails with email, password, and authorities
  - Maps user role to Spring Security authorities
- **Returns:** UserDetails object
- **Throws:** UsernameNotFoundException if user not found

---

## Data Models

### User Entity
- **Fields:**
  - id: Primary key
  - name: User's full name
  - email: Unique email address
  - password: BCrypt encoded password
  - role: ADMIN or USER
  - phone: Contact number
  - address: Physical address
  - profilePhoto: Filename of profile photo
  - stableId: Unique ID for face recognition
  - createdAt: Timestamp of creation
  - updatedAt: Timestamp of last update
  - visiblePassword: Plain text password (for demo purposes)

### Attendance Entity
- **Fields:**
  - id: Primary key
  - user: Reference to User
  - stableId: Face recognition ID
  - eventType: "ENTRY" or "EXIT"
  - timestamp: Date and time of event
  - confidence: Recognition confidence score

### Leave Entity
- **Fields:**
  - id: Primary key
  - user: Reference to User
  - leaveType: Type of leave
  - startDate: Leave start date
  - endDate: Leave end date
  - reason: Leave reason
  - status: PENDING, APPROVED, or REJECTED
  - createdAt: Application timestamp

### Asset Entity
- **Fields:**
  - id: Primary key
  - assetName: Name of asset
  - assetType: Type/category of asset
  - status: AVAILABLE, REQUESTED, or ASSIGNED
  - assignedUser: Reference to User (if assigned)
  - createdAt: Creation timestamp

---

## Configuration & Security

### Security Configuration
- **JWT-based authentication**
- **Role-based access control (RBAC)**
- **Password encoding with BCrypt**
- **CORS configuration for frontend integration**
- **Public endpoints:** `/api/auth/login`, `/api/auth/register`
- **Protected endpoints:** All others require authentication
- **Admin-only endpoints:** Marked with `@PreAuthorize("hasRole('ADMIN')")`

### Database Configuration
- **Primary Database:** PostgreSQL (via JPA/Hibernate)
- **Secondary Database:** SQLite (for AI script integration)
- **Custom SQLite Dialect:** For compatibility
- **Connection pooling:** Managed by Spring Boot

### File Upload Configuration
- **Profile photos:** `./uploads/profiles/`
- **Video uploads:** `../AI_Portion/uploads/`
- **UUID-based filenames** for uniqueness

---

## Integration Points

### AI Script Integration
- **SQLite Database:** Shared database for face embeddings and attendance
- **Video Processing:** Asynchronous Python script execution
- **User Sync:** Automatic user creation from face_embeddings
- **Attendance Sync:** Periodic sync from SQLite to PostgreSQL

### Frontend Integration
- **RESTful API:** JSON request/response format
- **JWT Authentication:** Token-based stateless authentication
- **CORS Enabled:** Cross-origin requests allowed
- **File Upload:** Multipart form data support

---

## Error Handling

### Global Exception Handler
- **ResourceNotFoundException:** Returns 404 Not Found
- **Authentication Errors:** Returns 401 Unauthorized
- **Authorization Errors:** Returns 403 Forbidden
- **Validation Errors:** Returns 400 Bad Request
- **Generic Errors:** Returns 500 Internal Server Error

### Logging
- **SLF4J with Logback**
- **Log levels:** INFO for operations, ERROR for failures, DEBUG for details
- **Logged events:** Login attempts, registration, sync operations, video processing

---

## Summary

The backend provides a comprehensive REST API for:
1. **User authentication and authorization** with JWT
2. **User profile management** with photo uploads
3. **Attendance tracking** with AI integration
4. **Leave management** with approval workflow
5. **Asset management** with request/assign workflow
6. **Admin operations** including video processing and data synchronization

All endpoints are secured with role-based access control, and the system integrates seamlessly with the AI-powered face recognition script through shared SQLite database and asynchronous video processing.
