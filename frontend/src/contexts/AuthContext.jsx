import { createContext, useContext, useState, useEffect } from 'react';
import apiService from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedToken = localStorage.getItem('auth_token');
    const storedUser = localStorage.getItem('auth_user');
    
    if (storedToken && storedUser) {
      try {
        const userData = JSON.parse(storedUser);
        setToken(storedToken);
        setUser(userData);
        setIsAuthenticated(true);
        apiService.setAuthToken(storedToken);
      } catch (error) {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');
      }
    }
    setLoading(false);
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