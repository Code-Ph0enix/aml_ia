import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
})

// Add token to every request
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Handle 401 errors (token expired)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

// ============================================================================
// AUTH API
// ============================================================================

export const authAPI = {
  register: async (email, password, fullName) => {
    const response = await api.post('/api/v1/auth/register', {
      email,
      password,
      full_name: fullName
    })
    return response.data
  },

  login: async (email, password) => {
    const response = await api.post('/api/v1/auth/login', {
      email,
      password
    })
    return response.data
  },

  me: async () => {
    const response = await api.get('/api/v1/auth/me')
    return response.data
  },

  logout: async () => {
    const response = await api.post('/api/v1/auth/logout')
    return response.data
  }
}

// ============================================================================
// CHAT API
// ============================================================================

export const chatAPI = {
  sendMessage: async (query, conversationId = null) => {
    const response = await api.post('/api/v1/chat/', {
      query,
      conversation_id: conversationId
    })
    return response.data
  },

  getHistory: async (conversationId) => {
    const response = await api.get(`/api/v1/chat/history/${conversationId}`)
    return response.data
  },

  getConversations: async () => {
    const response = await api.get('/api/v1/chat/conversations')
    return response.data
  },

  deleteConversation: async (conversationId) => {
    const response = await api.delete(`/api/v1/chat/conversation/${conversationId}`)
    return response.data
  }
}

// Export default object with all APIs
export default {
  auth: authAPI,
  chat: chatAPI
}
