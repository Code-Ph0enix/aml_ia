// import { createContext, useContext, useState } from 'react'
// import { chatAPI } from '../services/api'

// const ChatContext = createContext()

// export const useChat = () => {
//   const context = useContext(ChatContext)
//   if (!context) {
//     throw new Error('useChat must be used within ChatProvider')
//   }
//   return context
// }

// export const ChatProvider = ({ children }) => {
//   const [messages, setMessages] = useState([])
//   const [isLoading, setIsLoading] = useState(false)
//   const [error, setError] = useState(null)
//   const [conversations, setConversations] = useState([])

//   const sendMessage = async (message, conversationId) => {
//     try {
//       setIsLoading(true)
//       setError(null)

//       // Add user message immediately
//       const userMessage = {
//         role: 'user',
//         content: message,
//         timestamp: new Date().toISOString(),
//       }
//       setMessages(prev => [...prev, userMessage])

//       // Call API
//       const response = await chatAPI.sendMessage(message, conversationId)

//       // Add assistant response
//       const assistantMessage = {
//         role: 'assistant',
//         content: response.response,
//         timestamp: new Date().toISOString(),
//         metadata: response.metadata || {},
//       }
//       setMessages(prev => [...prev, assistantMessage])

//       return response
//     } catch (err) {
//       setError(err.message || 'Failed to send message')
//       throw err
//     } finally {
//       setIsLoading(false)
//     }
//   }

//   const loadConversations = async () => {
//     try {
//       const data = await chatAPI.getConversations()
//       setConversations(data.conversations || [])
//     } catch (err) {
//       console.error('Failed to load conversations:', err)
//     }
//   }

//   const clearMessages = () => {
//     setMessages([])
//   }

//   return (
//     <ChatContext.Provider
//       value={{
//         messages,
//         isLoading,
//         error,
//         conversations,
//         sendMessage,
//         loadConversations,
//         clearMessages,
//       }}
//     >
//       {children}
//     </ChatContext.Provider>
//   )
// }








import { createContext, useContext, useState } from 'react'
import { chatAPI } from '../services/api'

const ChatContext = createContext()

export const useChat = () => {
  const context = useContext(ChatContext)
  if (!context) {
    throw new Error('useChat must be used within ChatProvider')
  }
  return context
}

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [conversations, setConversations] = useState([])

  const sendMessage = async (message, conversationId) => {
    try {
      setIsLoading(true)
      setError(null)

      // Add user message immediately
      const userMessage = {
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
      }
      setMessages(prev => [...prev, userMessage])

      // Call API
      const response = await chatAPI.sendMessage(message, conversationId)

      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString(),
        metadata: response.metadata || {},
      }
      setMessages(prev => [...prev, assistantMessage])

      return response
    } catch (err) {
      setError(err.message || 'Failed to send message')
      throw err
    } finally {
      setIsLoading(false)
    }
  }

  const loadConversations = async () => {
    try {
      const data = await chatAPI.getConversations()
      setConversations(data.conversations || [])
    } catch (err) {
      // Silently fail - endpoint not implemented yet
      console.log('Conversation history not available yet')
      setConversations([])
    }
  }

  const clearMessages = () => {
    setMessages([])
  }

  return (
    <ChatContext.Provider
      value={{
        messages,
        isLoading,
        error,
        conversations,
        sendMessage,
        loadConversations,
        clearMessages,
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}
