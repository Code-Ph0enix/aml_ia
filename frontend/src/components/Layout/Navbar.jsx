// import { motion } from 'framer-motion'
// import { HiMenu, HiX } from 'react-icons/hi'
// import { BsRobot } from 'react-icons/bs'
// import ThemeToggle from '../UI/ThemeToggle'

// const Navbar = ({ onToggleSidebar, sidebarOpen }) => {
//   return (
//     <motion.nav
//       initial={{ y: -20, opacity: 0 }}
//       animate={{ y: 0, opacity: 1 }}
//       className="flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm"
//     >
//       {/* Left: Menu + Logo */}
//       <div className="flex items-center gap-4">
//         <button
//           onClick={onToggleSidebar}
//           className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors lg:hidden"
//           aria-label="Toggle sidebar"
//         >
//           {sidebarOpen ? (
//             <HiX className="w-6 h-6 text-gray-700 dark:text-gray-300" />
//           ) : (
//             <HiMenu className="w-6 h-6 text-gray-700 dark:text-gray-300" />
//           )}
//         </button>
        
//         <div className="flex items-center gap-2">
//           <BsRobot className="w-6 h-6 text-blue-600 dark:text-blue-400" />
//           <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
//             Banking RAG AI
//           </h1>
//         </div>
//       </div>
      
//       {/* Right: Theme Toggle */}
//       <div className="flex items-center gap-2">
//         <ThemeToggle />
//       </div>
//     </motion.nav>
//   )
// }

// export default Navbar







import { HiMenuAlt2, HiX, HiMoon, HiSun, HiLogout, HiUser } from 'react-icons/hi'
import { useState } from 'react'
import { useAuth } from '../../context/AuthContext'
import Button from '../UI/Button'

const Navbar = ({ onToggleSidebar, darkMode, onToggleDarkMode }) => {
  const { user, logout } = useAuth()
  const [showUserMenu, setShowUserMenu] = useState(false)

  return (
    <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3">
      <div className="flex items-center justify-between">
        {/* Left: Hamburger + Logo */}
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleSidebar}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors lg:hidden"
          >
            <HiMenuAlt2 className="w-6 h-6 text-gray-700 dark:text-gray-300" />
          </button>

          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center">
              <span className="text-white font-bold text-sm">Q</span>
            </div>
            <span className="font-bold text-xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent hidden sm:block">
              QuestRAG AI
            </span>
          </div>
        </div>

        {/* Right: User Info + Dark Mode Toggle */}
        <div className="flex items-center gap-3">
          {/* Dark Mode Toggle */}
          <button
            onClick={onToggleDarkMode}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label="Toggle dark mode"
          >
            {darkMode ? (
              <HiSun className="w-5 h-5 text-yellow-500" />
            ) : (
              <HiMoon className="w-5 h-5 text-gray-600" />
            )}
          </button>

          {/* User Menu */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                <HiUser className="w-5 h-5 text-white" />
              </div>
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300 hidden md:block">
                {user?.full_name || user?.email || 'User'}
              </span>
            </button>

            {/* Dropdown Menu */}
            {showUserMenu && (
              <>
                {/* Backdrop */}
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setShowUserMenu(false)}
                />

                {/* Menu */}
                <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-20">
                  <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                    <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                      {user?.full_name}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                      {user?.email}
                    </p>
                  </div>

                  <div className="p-2">
                    <button
                      onClick={() => {
                        setShowUserMenu(false)
                        logout()
                      }}
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                    >
                      <HiLogout className="w-4 h-4" />
                      Logout
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar

