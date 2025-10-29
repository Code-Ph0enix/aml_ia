import { createContext, useContext, useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../services/api'

const AuthContext = createContext(null)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(localStorage.getItem('token'))
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  // Check if user is logged in on mount
  useEffect(() => {
    if (token) {
      // Verify token and get user info
      api.auth.me()
        .then(userData => {
          setUser(userData)
        })
        .catch(() => {
          // Token invalid, clear it
          logout()
        })
        .finally(() => {
          setLoading(false)
        })
    } else {
      setLoading(false)
    }
  }, [token])

  const login = async (email, password) => {
    try {
      const response = await api.auth.login(email, password)
      const { access_token, user: userData } = response
      
      // Save token and user
      localStorage.setItem('token', access_token)
      setToken(access_token)
      setUser(userData)
      
      // Navigate to chat
      navigate('/chat')
      
      return { success: true }
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Login failed' 
      }
    }
  }

  const signup = async (email, password, fullName) => {
    try {
      const response = await api.auth.register(email, password, fullName)
      const { access_token, user: userData } = response
      
      // Save token and user
      localStorage.setItem('token', access_token)
      setToken(access_token)
      setUser(userData)
      
      // Navigate to chat
      navigate('/chat')
      
      return { success: true }
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Signup failed' 
      }
    }
  }

  const logout = () => {
    localStorage.removeItem('token')
    setToken(null)
    setUser(null)
    navigate('/login')
  }

  const value = {
    user,
    token,
    loading,
    login,
    signup,
    logout,
    isAuthenticated: !!token && !!user
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}
