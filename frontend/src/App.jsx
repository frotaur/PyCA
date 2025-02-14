import './App.css'
import World from './World'
import Baricelli1D from './models/barricelli'
import CA2D from './models/ca2d'
import { useState } from 'react'

function App() {
  const [selectedModel, setSelectedModel] = useState('barricelli')

  return (
    <div className="App">
      <div className="model-selector">
        <select 
          value={selectedModel} 
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          <option value="barricelli">Barricelli 1D</option>
          <option value="gol">Game of Life</option>
        </select>
      </div>

      <h1>{selectedModel === 'barricelli' ? 'Baricelli 1D Cellular Automaton' : 'Game of Life'}</h1>
      
      {selectedModel === 'barricelli' ? (
        <Baricelli1D width={800} height={400} nSpecies={6} />
      ) : (
        <CA2D width={400} height={400} cellSize={2} />
      )}
    </div>
  )
}

export default App