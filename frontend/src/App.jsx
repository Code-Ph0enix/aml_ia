import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useState } from 'react'
import { AuthProvider } from './context/AuthContext'
import { ChatProvider } from './context/ChatContext'
import { ThemeProvider } from './context/ThemeContext'
import ProtectedRoute from './components/Auth/ProtectedRoute'
import Login from './components/Auth/Login'
import Signup from './components/Auth/Signup'
import Navbar from './components/Layout/Navbar'
import Sidebar from './components/Layout/Sidebar'
import ChatContainer from './components/Chat/ChatContainer'
import WelcomeScreen from './components/Layout/WelcomeScreen'

// Main Chat Page Component (your original structure)
function ChatPage() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [currentConversationId, setCurrentConversationId] = useState(null)

  return (
    <ThemeProvider>
      <ChatProvider>
        <div className="flex h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
          {/* Sidebar */}
          <Sidebar 
            isOpen={sidebarOpen}
            onClose={() => setSidebarOpen(false)}
            onSelectConversation={setCurrentConversationId}
            currentConversationId={currentConversationId}
          />

          {/* Main Content */}
          <div className="flex flex-col flex-1 overflow-hidden">
            {/* Navbar */}
            <Navbar 
              onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
              sidebarOpen={sidebarOpen}
            />

            {/* Chat Area */}
            <main className="flex-1 overflow-hidden">
              {currentConversationId ? (
                <ChatContainer conversationId={currentConversationId} />
              ) : (
                <WelcomeScreen onNewChat={() => setCurrentConversationId('new')} />
              )}
            </main>
          </div>
        </div>
      </ChatProvider>
    </ThemeProvider>
  )
}

// Main App with Auth Routing
function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />

          {/* Protected Route - Your Original Chat */}
          <Route
            path="/chat"
            element={
              <ProtectedRoute>
                <ChatPage />
              </ProtectedRoute>
            }
          />

          {/* Redirect root to chat */}
          <Route path="/" element={<Navigate to="/chat" replace />} />

          {/* 404 - Redirect to chat */}
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}

export default App






// import { useState } from 'react'
// import { ChatProvider } from './context/ChatContext'
// import { ThemeProvider } from './context/ThemeContext'
// import Navbar from './components/Layout/Navbar'
// import Sidebar from './components/Layout/Sidebar'
// import ChatContainer from './components/Chat/ChatContainer'
// import WelcomeScreen from './components/Layout/WelcomeScreen'

// function App() {
//   const [sidebarOpen, setSidebarOpen] = useState(true)
//   const [currentConversationId, setCurrentConversationId] = useState(null)

//   return (
//     <ThemeProvider>
//       <ChatProvider>
//         <div className="flex h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
//           {/* Sidebar */}
//           <Sidebar 
//             isOpen={sidebarOpen}
//             onClose={() => setSidebarOpen(false)}
//             onSelectConversation={setCurrentConversationId}
//             currentConversationId={currentConversationId}
//           />

//           {/* Main Content */}
//           <div className="flex flex-col flex-1 overflow-hidden">
//             {/* Navbar */}
//             <Navbar 
//               onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
//               sidebarOpen={sidebarOpen}
//             />

//             {/* Chat Area */}
//             <main className="flex-1 overflow-hidden">
//               {currentConversationId ? (
//                 <ChatContainer conversationId={currentConversationId} />
//               ) : (
//                 <WelcomeScreen onNewChat={() => setCurrentConversationId('new')} />
//               )}
//             </main>
//           </div>
//         </div>
//       </ChatProvider>
//     </ThemeProvider>
//   )
// }

// export default App