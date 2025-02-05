import { useState, useEffect, useRef } from 'react';
import { vec2 } from 'three/tsl';

function createCanvas(data, width, height) {
    if (width*height !== data.length) throw new Error("image size and array shape not compatible")
    let canvas = document.createElement("canvas")
    canvas.width = width 
    canvas.height = height 
    let ctx = canvas.getContext("2d")
    let imgData = ctx.createImageData(width, height)
    for (let i = 0; i < data.length; i++) {
        imgData.data[i*4+0] = data[i][0];
        imgData.data[i*4+1] = data[i][1];
        imgData.data[i*4+2] = data[i][2];
        imgData.data[i*4+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas
}

function World() {

    const [data, setData] = useState(null);
    const canvasRef = useRef(null);

    useEffect(() => {
        const sse = new EventSource("http://localhost:5000/stream");
        function handleStream(e) {
            const jsonStr = e.data.replace("data: ", "")
            const frameData = JSON.parse(jsonStr)
            setData(frameData)
        }
        sse.onmessage = e => {
            handleStream(e)
        }        
        sse.onerror = e => {
            sse.close()
        }
        return () => {
            sse.close();
        }
    }, )

    useEffect(() => {
        if (!data || !canvasRef.current) return; 
        
        const displayCtx = canvasRef.current.getContext("2d");

        const imageCanvas = createCanvas(
            data.frame, 
            data.dimensions.width, 
            data.dimensions.height
        )

        displayCtx.drawImage(imageCanvas, 0, 0);

    }, [data]);
    
    return (
        <canvas
            ref = {canvasRef}
            width = {300}
            height = {300}
            sstyle = {{border: "1px solid black"}}
        />
    );
}

export default World
        
            