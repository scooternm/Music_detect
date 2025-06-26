import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import MusicSimilarityApp from './MusicSimilarityApp.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <MusicSimilarityApp />
  </StrictMode>,
)
