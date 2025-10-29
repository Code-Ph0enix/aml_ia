import { useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { HiPlus, HiChat, HiTrash } from 'react-icons/hi'
import { useChat } from '../../context/ChatContext'
import Button from '../UI/Button'

const Sidebar = ({ isOpen, onClose, onSelectConversation, currentConversationId }) => {
  const { conversations, loadConversations } = useChat()
  
  useEffect(() => {
    loadConversations()
  }, [])
  
  const sidebarVariants = {
    open: { x: 0 },
    closed: { x: '-100%' },
  }
  
  return (
    <>
      {/* Backdrop for mobile */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          />
        )}
      </AnimatePresence>
      
      {/* Sidebar */}
      <motion.aside
        initial="closed"
        animate={isOpen ? 'open' : 'closed'}
        variants={sidebarVariants}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="fixed lg:relative inset-y-0 left-0 z-50 w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col shadow-lg lg:shadow-none"
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <Button
            variant="primary"
            className="w-full"
            icon={<HiPlus className="w-5 h-5" />}
            onClick={() => {
              onSelectConversation('new')
              onClose()
            }}
          >
            New Chat
          </Button>
        </div>
        
        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {conversations.length === 0 ? (
            <div className="text-center text-gray-500 dark:text-gray-400 mt-8">
              <HiChat className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No conversations yet</p>
            </div>
          ) : (
            conversations.map((conv) => (
              <motion.button
                key={conv.conversation_id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => {
                  onSelectConversation(conv.conversation_id)
                  onClose()
                }}
                className={`w-full p-3 rounded-lg text-left transition-colors ${
                  currentConversationId === conv.conversation_id
                    ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {conv.title || 'New Conversation'}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                      {new Date(conv.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      // Delete conversation
                    }}
                    className="p-1 hover:bg-red-100 dark:hover:bg-red-900/30 rounded"
                  >
                    <HiTrash className="w-4 h-4 text-red-500" />
                  </button>
                </div>
              </motion.button>
            ))
          )}
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
            Powered by RAG + RL
          </p>
        </div>
      </motion.aside>
    </>
  )
}

export default Sidebar
