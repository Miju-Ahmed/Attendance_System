import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { Login } from './components/auth/Login';
import { Register } from './components/auth/Register';
import { ProtectedRoute } from './components/auth/ProtectedRoute';
import { UserDashboard } from './components/user/Dashboard';
import { Profile } from './components/user/Profile';
import { AssetApplication } from './components/user/AssetApplication';
import { LeaveApplication } from './components/user/LeaveApplication';
import { ApplicationStatus } from './components/user/ApplicationStatus';
import { AdminDashboard } from './components/admin/AdminDashboard';
import { UserList } from './components/admin/UserList';
import { UserDetails } from './components/admin/UserDetails';
import { LeaveApproval } from './components/admin/LeaveApproval';
import { AttendanceOverview } from './components/admin/AttendanceOverview';

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* User Routes */}
          <Route
            path="/user/dashboard"
            element={
              <ProtectedRoute>
                <UserDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/user/profile"
            element={
              <ProtectedRoute>
                <Profile />
              </ProtectedRoute>
            }
          />
          <Route
            path="/user/assets"
            element={
              <ProtectedRoute>
                <AssetApplication />
              </ProtectedRoute>
            }
          />
          <Route
            path="/user/leaves"
            element={
              <ProtectedRoute>
                <LeaveApplication />
              </ProtectedRoute>
            }
          />
          <Route
            path="/user/status"
            element={
              <ProtectedRoute>
                <ApplicationStatus />
              </ProtectedRoute>
            }
          />
          <Route
            path="/user/attendance"
            element={
              <ProtectedRoute>
                <AttendanceOverview />
              </ProtectedRoute>
            }
          />

          {/* Admin Routes */}
          <Route
            path="/admin/dashboard"
            element={
              <ProtectedRoute adminOnly>
                <AdminDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/users"
            element={
              <ProtectedRoute adminOnly>
                <UserList />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/users/:userId"
            element={
              <ProtectedRoute adminOnly>
                <UserDetails />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/leaves"
            element={
              <ProtectedRoute adminOnly>
                <LeaveApproval />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/attendance"
            element={
              <ProtectedRoute adminOnly>
                <AttendanceOverview />
              </ProtectedRoute>
            }
          />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
