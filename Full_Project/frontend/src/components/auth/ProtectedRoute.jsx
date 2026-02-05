import { Navigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

export const ProtectedRoute = ({ children, adminOnly = false }) => {
    const { isAuthenticated, isAdmin } = useAuth();

    if (!isAuthenticated) {
        return <Navigate to="/" replace />;
    }

    if (adminOnly && !isAdmin) {
        return <Navigate to="/user/dashboard" replace />;
    }

    return children;
};
