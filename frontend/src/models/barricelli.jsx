import React, { useEffect, useRef, useState } from 'react';

class Baricelli1DAutomaton {
    constructor(width, height, nSpecies = 6, reprodCollision = false) {
        this.width = width;
        this.height = height;
        this.time = 0;
        this.speciesNum = nSpecies;
        this.repCol = reprodCollision;
        
        // Create position array [0, 1, 2, ..., width-1]
        this.positions = Array.from({ length: width }, (_, i) => i);
        
        // Initialize random world state
        this.world = Array.from({ length: width }, () => 
            Math.floor(Math.random() * (2 * nSpecies + 1)) - nSpecies
        );
    }

    step() {
        const newWorld = new Array(this.width).fill(0);

        // First, handle movement
        const nonZeroIndices = this.world.map((val, idx) => val !== 0 ? idx : -1).filter(idx => idx !== -1);
        const movers = nonZeroIndices.map(idx => this.world[idx]);
        const targetPositions = nonZeroIndices.map(idx => 
            (this.positions[idx] + this.world[idx]) % this.width
        );

        // Handle movement collisions
        const [moveMask, moveLocs] = this._moveCollision(targetPositions);
        moveLocs.forEach((loc, i) => {
            if (moveMask[i]) {
                newWorld[loc] = movers[i];
            }
        });

        // Handle reproduction
        const reproduceAttempts = this.world.map((val, idx) => 
            val !== 0 && newWorld[idx] !== 0 ? idx : -1
        ).filter(idx => idx !== -1);

        const reprodParents = reproduceAttempts.map(idx => newWorld[idx]);
        const repPositions = reproduceAttempts.map(idx => 
            (this.positions[idx] + this.world[idx] - newWorld[idx]) % this.width
        );

        const [repMask, repLocs] = this._moveCollision(repPositions);
        
        repLocs.forEach((loc, i) => {
            if (repMask[i]) {
                const parent = reprodParents[i];
                if (newWorld[loc] === 0 || newWorld[loc] === parent) {
                    newWorld[loc] = parent;
                } else if (this.repCol) {
                    newWorld[loc] = 0;
                }
            }
        });

        this.time++;
        this.world = newWorld;
    }

    _moveCollision(targetPositions) {
        // Count occurrences of each target position
        const counts = new Array(this.width).fill(0);
        targetPositions.forEach(pos => counts[pos]++);

        // Create success mask for positions with only one particle targeting them
        const canMove = counts.map(count => count === 1);
        const successMask = targetPositions.map(pos => canMove[pos]);
        const successMoves = targetPositions.filter((_, i) => successMask[i]);

        return [successMask, successMoves];
    }

    getColorWorld() {
        return this.world.map(val => {
            const h = val / (this.speciesNum * 2) + 0.5;
            const s = 0.7;
            const v = val === 0 ? 0 : 0.8;
            return hsvToRgb(h, s, v);
        });
    }

    reset() {
        this.time = 0;
        this.world = Array.from({ length: this.width }, () => 
            Math.floor(Math.random() * (2 * this.speciesNum + 1)) - this.speciesNum
        );
    }
}

// Helper function to convert HSV to RGB
function hsvToRgb(h, s, v) {
    let r, g, b;
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }

    return [
        Math.round(r * 255),
        Math.round(g * 255),
        Math.round(b * 255)
    ];
}

const Baricelli1D = ({ width = 800, height = 400, nSpecies = 6 }) => {
    const canvasRef = useRef(null);
    const automaton = useRef(null);
    const animationFrame = useRef(null);
    const [isRunning, setIsRunning] = useState(true);
    const [reproductionCollision, setReproductionCollision] = useState(false);

    useEffect(() => {
        automaton.current = new Baricelli1DAutomaton(width, height, nSpecies, reproductionCollision);
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        const animate = () => {
            if (isRunning) {
                automaton.current.step();
                const colors = automaton.current.getColorWorld();
                
                // Draw the current state
                const imageData = ctx.createImageData(width, 1);
                colors.forEach((color, i) => {
                    const idx = i * 4;
                    imageData.data[idx] = color[0];     // R
                    imageData.data[idx + 1] = color[1]; // G
                    imageData.data[idx + 2] = color[2]; // B
                    imageData.data[idx + 3] = 255;      // A
                });

                // Copy previous frame down
                ctx.drawImage(canvas, 0, 0, width, height - 1, 0, 1, width, height - 1);
                // Draw new line at top
                ctx.putImageData(imageData, 0, 0);
            }
            animationFrame.current = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            if (animationFrame.current) {
                cancelAnimationFrame(animationFrame.current);
            }
        };
    }, [width, height, nSpecies, isRunning, reproductionCollision]);

    const handleKeyDown = (e) => {
        switch (e.key) {
            case 'Delete':
            case 'Backspace':
                automaton.current.reset();
                break;
            case 'ArrowUp':
                automaton.current.speciesNum = Math.min(20, automaton.current.speciesNum + 1);
                break;
            case 'ArrowDown':
                automaton.current.speciesNum = Math.max(1, automaton.current.speciesNum - 1);
                break;
            case 'g':
                setReproductionCollision(!reproductionCollision);
                break;
            case ' ':
                setIsRunning(!isRunning);
                break;
            default:
                break;
        }
    };

    return (
        <div onKeyDown={handleKeyDown} tabIndex={0} style={{ outline: 'none' }}>
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                style={{ border: '1px solid #000' }}
            />
            <div style={{ marginTop: '10px' }}>
                <button onClick={() => setIsRunning(!isRunning)}>
                    {isRunning ? 'Pause' : 'Play'}
                </button>
                <button onClick={() => automaton.current.reset()}>Reset</button>
                <button onClick={() => setReproductionCollision(!reproductionCollision)}>
                    Toggle Reproduction Collision
                </button>
            </div>
        </div>
    );
};

export default Baricelli1D;