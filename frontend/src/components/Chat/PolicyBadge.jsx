import { motion } from 'framer-motion'
import { HiDatabase, HiX } from 'react-icons/hi'

const PolicyBadge = ({ action, confidence }) => {
  const isFetch = action === 'FETCH'
  
  return (
    <motion.div
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ delay: 0.2 }}
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
        isFetch
          ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
          : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
      }`}
    >
      {isFetch ? (
        <HiDatabase className="w-3 h-3" />
      ) : (
        <HiX className="w-3 h-3" />
      )}
      <span>{action}</span>
      {confidence && (
        <span className="ml-1 opacity-75">
          ({(confidence * 100).toFixed(0)}%)
        </span>
      )}
    </motion.div>
  )
}

export default PolicyBadge
