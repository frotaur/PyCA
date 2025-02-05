import { useState, useEffect } from 'react';

function World() {
    const [frame, setFrame] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const eventSource = new EventSource("http://localhost:5000/stream");
        
        eventSource.onopen = () => {
            setIsConnected(true);
            setError(null);
            console.log("SSE connection opened");
        };

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setFrame(data.frame);
            } catch (err) {
                console.error("Failed to parse message:", err);
            }
        };

        eventSource.onerror = (error) => {
            console.error("SSE error:", error);
            setIsConnected(false);
            setError("Connection failed");
        };

        return () => eventSource.close();
    }, []);
    
    return (
        <div>
            {error && <div style={{color: 'red'}}>{error}</div>}
            {isConnected ? 'Connected to stream' : 'Waiting for connection...'}
            <br/>
            {frame ? 'Receiving frames...' : 'No frames yet'}
        </div>
    );
}

export default World
        
            