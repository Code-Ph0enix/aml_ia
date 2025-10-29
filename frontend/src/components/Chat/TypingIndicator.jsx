import { motion } from 'framer-motion'

const TypingIndicator = () => {
  return (
    <div className="flex items-center gap-2 p-4 max-w-xs">
      <div className="flex items-center gap-1 p-3 rounded-2xl bg-gray-200 dark:bg-gray-700">
        {[0, 1, 2].map((index) => (
          <motion.div
            key={index}
            className="w-2 h-2 bg-gray-500 dark:bg-gray-400 rounded-full"
            animate={{
              y: [0, -8, 0],
            }}
            transition={{
              duration: 0.6,
              repeat: Infinity,
              delay: index * 0.15,
            }}
          />
        ))}
      </div>
      <span className="text-sm text-gray-500 dark:text-gray-400">
        AI is thinking...
      </span>
    </div>
  )
}

export default TypingIndicator
