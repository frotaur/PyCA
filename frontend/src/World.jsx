import { useState, useEffect, useRef, useCallback } from 'react';
import pako from "pako"
import pickleparser from "pickleparser"


function World() {

    const [height, setHeight] = useState(null);
    const [width, setWidth] = useState(null);
    const [channels, setChannels] = useState(null);
    const [error, setError] = useState(null);
    const [fps, setFps] = useState(0);
    
    // Refs for performance
    const frameCount = useRef(0);
    const lastTime = useRef(performance.now());
    const animationFrameId = useRef(null);
    const parser = useRef(new pickleparser.Parser());
    const canvasRef = useRef(null);
    const ctxRef = useRef(null);
    const imageDataRef = useRef(null);

    // Create refs for animation state management
    const isStreaming = useRef(false);

    // Add this ref at the top with other refs
    const initialFrameData = useRef(null);

    const initCanvas = () => {
        if (!width || !height || !canvasRef.current) return;

        ctxRef.current = canvasRef.current.getContext("2d", { alpha: false })
        imageDataRef.current = ctxRef.current.createImageData(height, width)
    }

    const updateCanvas = (data) => {

        if (!ctxRef.current || !imageDataRef.current) return;

        for (let i = 0, j = 0; i < data.length; i += 3, j += 4) {
            imageDataRef.current.data[j+0] = data[i+0];
            imageDataRef.current.data[j+1] = data[i+1]; 
            imageDataRef.current.data[j+2] = data[i+2];
            imageDataRef.current.data[j+3] = 255;
        }

        ctxRef.current.putImageData(imageDataRef.current, 0, 0);
    }


    const fetchStream = async () => {
        try {
            const response = await fetch("http://localhost:5000/stream");
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const arrayBuffer = await response.arrayBuffer();
            const encodedData = pako.inflate(arrayBuffer);
            const data = parser.current.parse(encodedData);

            // Only update dimensions if they're not set or different
            if (!width || !height || !channels) {
                setWidth(Math.sqrt(data[4].length / 3));
                setHeight(Math.sqrt(data[4].length / 3));
                setChannels(3);
            }

            updateCanvas(data[4]);
            
            // Update FPS counter
            frameCount.current++;
            const now = performance.now();
            if (now - lastTime.current >= 1000) {
                setFps(frameCount.current);
                frameCount.current = 0;
                lastTime.current = now;
            }

        } catch (error) {
            console.error("Stream error:", error);
        }

        // Continue the animation loop if still streaming
        if (isStreaming.current) {
            animationFrameId.current = requestAnimationFrame(fetchStream);
        }
    };

    const startStream = () => {
        if (!isStreaming.current) {
            console.log("Starting stream");
            isStreaming.current = true;
            animationFrameId.current = requestAnimationFrame(fetchStream);
        }
    };

    const stopStream = () => {
        if (isStreaming.current) {
            console.log("Stopping stream");
            isStreaming.current = false;
            if (animationFrameId.current) {
                cancelAnimationFrame(animationFrameId.current);
                animationFrameId.current = null;
            }
        }
    };

    const fetchSingleFrame = async () => {
        try {
            const response = await fetch("http://localhost:5000/stream");
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const arrayBuffer = await response.arrayBuffer();
            const encodedData = pako.inflate(arrayBuffer);
            const data = parser.current.parse(encodedData);

            updateCanvas(data[4]);
        } catch (error) {
            console.error("Single frame fetch error:", error);
        }
    };

    const handleKeyPress = async (event) => {
        console.log(event.code)

        const key = event.code;

        if (!key) {
            console.log('Key not mapped, returning');
            return;
        }
        
        if (key == "Space") {
            event.preventDefault();
        }

        try {
            // Handle space key streaming toggle specially
            if (event.code === 'Space') {
                if (isStreaming.current) {
                    stopStream();
                } else {
                    startStream();
                }
            }
            const response = await fetch(`http://localhost:5000/keypress?key=${key}`, {
                method: "GET",
                headers: {
                    'Accept': 'application/json',
                }
            });            
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP error! keypress failed: ${response.status}`);
            }

            // Fetch a single frame if we're not streaming
            if (!isStreaming.current) {
                await fetchSingleFrame();
            }
        } catch (error) {
            console.error("Error sending keypress:", error);
        }
    };

    // Make sure the event listener is properly set up
    useEffect(() => {
        console.log('Setting up keypress listener'); // Debug log 6
        window.addEventListener("keydown", handleKeyPress);
        
        return () => {
            console.log('Removing keypress listener'); // Debug log 7
            window.removeEventListener("keydown", handleKeyPress);
            stopStream();
        };
    }, []); // Empty dependency array

    // Initial mount effect to start streaming
    useEffect(() => {
        // Fetch one frame immediately to initialize the canvas
        const initializeStream = async () => {
            try {
                const response = await fetch("http://localhost:5000/stream");
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const arrayBuffer = await response.arrayBuffer();
                const encodedData = pako.inflate(arrayBuffer);
                const data = parser.current.parse(encodedData);

                // Set initial dimensions
                setWidth(Math.sqrt(data[4].length / 3));
                setHeight(Math.sqrt(data[4].length / 3));
                setChannels(3);

                // Store the first frame data to render after canvas is initialized
                initialFrameData.current = data[4];
            } catch (error) {
                console.error("Initial stream error:", error);
                setError(error.message);
            }
        };

        initializeStream();
    }, []); // Run once on mount

    useEffect(() => {
        initCanvas();
        if (width && height && channels) {
            // If we have initial frame data, render it
            if (initialFrameData.current) {
                updateCanvas(initialFrameData.current);
            }
            // Start the stream
        }
    }, [width, height, channels]);

    return (
        <div className="world-container">
            {error ? (
                <div className="error-message">Error: {error}</div>
            ) : (
                <>
                    <canvas
                        ref={canvasRef}
                        width={width || 0}
                        height={height || 0}
                        className="world-canvas"
                        style={{
                            border: '1px solid black',
                            imageRendering: 'pixelated',
                            transform: 'rotate(90deg)'
                        }}
                    />
                    <div className="fps-counter">FPS: {fps}</div>
                </>
            )}
        </div>
    );
}

export default World
        
            
            