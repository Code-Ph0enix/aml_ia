// import { motion } from 'framer-motion'
// import { BsRobot } from 'react-icons/bs'
// import { HiSparkles, HiLightningBolt, HiShieldCheck } from 'react-icons/hi'
// import Button from '../UI/Button'

// const WelcomeScreen = ({ onNewChat }) => {
//   const features = [
//     {
//       icon: <HiSparkles className="w-6 h-6" />,
//       title: 'RAG-Powered',
//       description: 'Retrieval-augmented generation for accurate banking information'
//     },
//     {
//       icon: <HiLightningBolt className="w-6 h-6" />,
//       title: 'RL-Enhanced',
//       description: 'Smart policy network decides when to fetch documents'
//     },
//     {
//       icon: <HiShieldCheck className="w-6 h-6" />,
//       title: 'Secure & Private',
//       description: 'Your conversations are encrypted and stored securely'
//     }
//   ]

//   const exampleQueries = [
//     'What is my account balance?',
//     'How do I open a savings account?',
//     'What are the interest rates for fixed deposits?',
//     'How can I apply for a credit card?'
//   ]

//   return (
//     <div className="flex flex-col items-center justify-center h-full p-8 text-center">
//       <motion.div
//         initial={{ scale: 0.8, opacity: 0 }}
//         animate={{ scale: 1, opacity: 1 }}
//         transition={{ duration: 0.5 }}
//         className="max-w-3xl space-y-8"
//       >
//         {/* Logo & Title */}
//         <div className="space-y-4">
//           <motion.div
//             animate={{ rotate: [0, 10, -10, 0] }}
//             transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
//           >
//             <BsRobot className="w-20 h-20 mx-auto text-blue-600 dark:text-blue-400" />
//           </motion.div>
//           <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
//             QuestRAG
//           </h1>
//           <p className="text-lg text-gray-600 dark:text-gray-400">
//             Your intelligent banking companion powered by AI
//           </p>
//         </div>

//         {/* Features */}
//         <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
//           {features.map((feature, index) => (
//             <motion.div
//               key={index}
//               initial={{ y: 20, opacity: 0 }}
//               animate={{ y: 0, opacity: 1 }}
//               transition={{ delay: index * 0.1 + 0.3 }}
//               className="p-4 rounded-xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow"
//             >
//               <div className="flex flex-col items-center gap-2">
//                 <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400">
//                   {feature.icon}
//                 </div>
//                 <h3 className="font-semibold text-gray-900 dark:text-white">
//                   {feature.title}
//                 </h3>
//                 <p className="text-sm text-gray-600 dark:text-gray-400">
//                   {feature.description}
//                 </p>
//               </div>
//             </motion.div>
//           ))}
//         </div>

//         {/* Example Queries */}
//         <div className="space-y-4">
//           <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
//             Try asking:
//           </h3>
//           <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
//             {exampleQueries.map((query, index) => (
//               <motion.button
//                 key={index}
//                 initial={{ x: -20, opacity: 0 }}
//                 animate={{ x: 0, opacity: 1 }}
//                 transition={{ delay: index * 0.1 + 0.6 }}
//                 whileHover={{ scale: 1.02 }}
//                 whileTap={{ scale: 0.98 }}
//                 onClick={onNewChat}
//                 className="p-3 text-left rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-sm text-gray-700 dark:text-gray-300 transition-colors"
//               >
//                 "{query}"
//               </motion.button>
//             ))}
//           </div>
//         </div>

//         {/* CTA Button */}
//         <motion.div
//           initial={{ y: 20, opacity: 0 }}
//           animate={{ y: 0, opacity: 1 }}
//           transition={{ delay: 0.8 }}
//         >
//           <Button
//             variant="primary"
//             size="lg"
//             onClick={onNewChat}
//             className="shadow-lg shadow-blue-500/50"
//           >
//             Start Chatting
//           </Button>
//         </motion.div>
//       </motion.div>
//     </div>
//   )
// }

// export default WelcomeScreen

import { motion } from 'framer-motion'
import { BsRobot } from 'react-icons/bs'
import { HiSparkles, HiLightningBolt, HiDatabase } from 'react-icons/hi'
import Button from '../UI/Button'

const WelcomeScreen = ({ onNewChat }) => {
  const features = [
    {
      icon: <HiSparkles className="w-6 h-6" />,
      title: 'RAG-Powered',
      description: 'Retrieval-augmented generation for accurate banking information'
    },
    {
      icon: <HiLightningBolt className="w-6 h-6" />,
      title: 'RL-Enhanced',
      description: 'Smart policy network decides when to fetch documents'
    },
    {
      icon: <HiDatabase className="w-6 h-6" />,
      title: 'Persistent Storage',
      description: 'Conversations stored in MongoDB Atlas cloud database'
    }
  ]

  const exampleQueries = [
    'What is my account balance?',
    'How do I open a savings account?',
    'What are the interest rates for fixed deposits?',
    'How can I apply for a credit card?'
  ]

  return (
    <div className="flex flex-col items-center justify-center h-full p-8 text-center">
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="max-w-3xl space-y-8"
      >
        {/* Logo & Title */}
        <div className="space-y-4">
          <motion.div
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
          >
            <BsRobot className="w-20 h-20 mx-auto text-blue-600 dark:text-blue-400" />
          </motion.div>
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            QuestRAG
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Your intelligent banking companion powered by AI
          </p>
        </div>

        {/* Features - WITH BIG SHADOWS! */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: index * 0.1 + 0.3 }}
              className="p-6 rounded-xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105 hover:-translate-y-1"
            >
              <div className="flex flex-col items-center gap-3">
                <div className="p-3 rounded-lg bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 shadow-md">
                  {feature.icon}
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  {feature.title}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
                  {feature.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Example Queries */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Try asking:
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {exampleQueries.map((query, index) => (
              <motion.button
                key={index}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: index * 0.1 + 0.6 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={onNewChat}
                className="p-3 text-left rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-sm text-gray-700 dark:text-gray-300 transition-colors shadow-sm hover:shadow-md"
              >
                "{query}"
              </motion.button>
            ))}
          </div>
        </div>

        {/* CTA Button */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <Button
            variant="primary"
            size="lg"
            onClick={onNewChat}
            className="shadow-lg shadow-blue-500/50 hover:shadow-blue-500/70"
          >
            Start Chatting
          </Button>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default WelcomeScreen
