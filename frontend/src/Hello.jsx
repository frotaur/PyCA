import './App.css'
import { useState, useEffect } from 'react'
function Hello() {
  
  const [data, setData] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch("http://localhost:5000/hello")
      if (response.ok) {
        const data = await response.json();
        setData(data)
      } else {
        console.error("Failed to fetch data")
      }
    }
    fetchData()
  }, [])
    
return <div>{data ? data.text : 'Loading...'}</div>

}

export default Hello