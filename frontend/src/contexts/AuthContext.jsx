import { createContext, useContext, useState, useEffect } from 'react';
import apiService from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const validateAndRestoreSession = async () => {
      const storedToken = localStorage.getItem('auth_token');
      if (!storedToken) {
        setLoading(false);
        return;
      }

      apiService.setAuthToken(storedToken);

      try {
        const userResponse = await apiService.request('/api/auth/me');
        if (userResponse && userResponse.username) {
          setUser({ username: userResponse.username, userId: userResponse.user_id });
          setToken(storedToken);
          setIsAuthenticated(true);
        } else {
          throw new Error('Invalid token');
        }
      } catch (error) {
        console.error('Token validation failed:', error);
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');
        setIsAuthenticated(false);
      } finally {
        setLoading(false);
      }
    };

    validateAndRestoreSession();
  }, []);

  const login = async (username, password) => {
    const response = await apiService.login(username, password);
    const { access_token, username: user_name, user_id } = response;
    
    setToken(access_token);
    setUser({ username: user_name, userId: user_id });
    setIsAuthenticated(true);
    
    localStorage.setItem('auth_token', access_token);
    localStorage.setItem('auth_user', JSON.stringify({ username: user_name, userId: user_id }));
    apiService.setAuthToken(access_token);
    
    return response;
  };

  const register = async (username, password, email) => {
    const response = await apiService.register(username, password, email);
    const { access_token, username: user_name, user_id } = response;
    
    setToken(access_token);
    setUser({ username: user_name, userId: user_id });
    setIsAuthenticated(true);
    
    localStorage.setItem('auth_token', access_token);
    localStorage.setItem('auth_user', JSON.stringify({ username: user_name, userId: user_id }));
    apiService.setAuthToken(access_token);
    
    return response;
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    setIsAuthenticated(false);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    apiService.setAuthToken(null);
  };

  const value = {
    user,
    token,
    isAuthenticated,
    loading,
    login,
    register,
    logout
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}