import { useState } from 'react'
import { motion } from 'framer-motion'
import { HiPaperAirplane } from 'react-icons/hi'
import Button from '../UI/Button'

const MessageInput = ({ onSend, disabled }) => {
  const [message, setMessage] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (message.trim() && !disabled) {
      onSend(message)
      setMessage('')
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <motion.form
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      onSubmit={handleSubmit}
      className="p-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700"
    >
      <div className="flex gap-2 max-w-4xl mx-auto">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask me anything about banking..."
          disabled={disabled}
          rows={1}
          className="flex-1 px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 resize-none"
          style={{ minHeight: '48px', maxHeight: '120px' }}
        />
        <Button
          type="submit"
          variant="primary"
          disabled={disabled || !message.trim()}
          className="px-4"
        >
          <HiPaperAirplane className="w-5 h-5" />
        </Button>
      </div>
      <p className="text-xs text-gray-500 dark:text-gray-400 text-center mt-2">
        Press Enter to send, Shift+Enter for new line
      </p>
    </motion.form>
  )
}

export default MessageInput
