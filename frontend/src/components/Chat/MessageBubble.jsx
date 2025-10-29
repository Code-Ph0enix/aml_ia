import { motion } from 'framer-motion'
import { HiUser } from 'react-icons/hi'
import { BsRobot } from 'react-icons/bs'
import ReactMarkdown from 'react-markdown'
import PolicyBadge from './PolicyBadge'

const MessageBubble = ({ message, index }) => {
  const isUser = message.role === 'user'
  const metadata = message.metadata || {}

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
        isUser
          ? 'bg-blue-600 text-white'
          : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
      }`}>
        {isUser ? (
          <HiUser className="w-5 h-5" />
        ) : (
          <BsRobot className="w-5 h-5" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex flex-col gap-2 max-w-[70%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Message Bubble */}
        <div className={`px-4 py-2 rounded-2xl ${
          isUser
            ? 'bg-blue-600 text-white rounded-tr-none'
            : 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-tl-none'
        }`}>
          <ReactMarkdown
            className="text-sm prose prose-sm dark:prose-invert max-w-none"
            components={{
              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
              ul: ({ children }) => <ul className="list-disc ml-4 mb-2">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal ml-4 mb-2">{children}</ol>,
              code: ({ children }) => (
                <code className="px-1 py-0.5 rounded bg-black/10 dark:bg-white/10 text-xs">
                  {children}
                </code>
              ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>

        {/* Metadata (for assistant messages) */}
        {!isUser && metadata.policy_action && (
          <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <PolicyBadge 
              action={metadata.policy_action} 
              confidence={metadata.policy_confidence}
            />
            {metadata.documents_retrieved > 0 && (
              <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-full">
                {metadata.documents_retrieved} docs retrieved
              </span>
            )}
            {metadata.response_time && (
              <span className="opacity-60">
                {metadata.response_time}ms
              </span>
            )}
          </div>
        )}

        {/* Timestamp */}
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {new Date(message.timestamp).toLocaleTimeString()}
        </span>
      </div>
    </motion.div>
  )
}

export default MessageBubble