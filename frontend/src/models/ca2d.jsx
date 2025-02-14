import React, { useEffect, useRef, useState } from 'react';

class CA2DAutomaton {
    constructor(width, height, sRule = "23", bRule = "3", random = false) {
        this.width = width;
        this.height = height;
        this.sNum = this.getRuleNumber(sRule);
        this.bNum = this.getRuleNumber(bRule);
        this.random = random;
        
        // Initialize world
        this.world = Array(height).fill().map(() => 
            Array(width).fill(0)
        );
        this.reset();
    }

    getRuleNumber(rule) {
        if (!rule) return 0;
        return rule.split('').reduce((acc, digit) => 
            acc + Math.pow(2, parseInt(digit)), 0
        );
    }

    getRuleString(num) {
        return Array(9).fill().map((_, i) => 
            ((num >> i) & 1) ? i.toString() : ''
        ).filter(x => x).join('');
    }

    reset() {
        if (this.random) {
            // Fill with random values
            const randPortion = 0.5;
            const randHeight = Math.floor(this.height * randPortion);
            const randWidth = Math.floor(this.width * randPortion);
            const startY = Math.floor((this.height - randHeight) / 2);
            const startX = Math.floor((this.width - randWidth) / 2);

            this.world = Array(this.height).fill().map(() => Array(this.width).fill(0));
            
            for (let y = startY; y < startY + randHeight; y++) {
                for (let x = startX; x < startX + randWidth; x++) {
                    this.world[y][x] = Math.random() > 0.5 ? 1 : 0;
                }
            }
        } else {
            // Place small random square in center
            this.world = Array(this.height).fill().map(() => Array(this.width).fill(0));
            const centerY = Math.floor(this.height / 2);
            const centerX = Math.floor(this.width / 2);
            
            for (let y = -1; y <= 0; y++) {
                for (let x = -1; x <= 0; x++) {
                    this.world[centerY + y][centerX + x] = Math.random() > 0.5 ? 1 : 0;
                }
            }
        }
    }

    getNthBit(num, n) {
        return (num >> n) & 1;
    }

    step() {
        const newWorld = Array(this.height).fill().map(() => Array(this.width).fill(0));
        
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                // Count neighbors (with wraparound)
                let count = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        
                        const ny = (y + dy + this.height) % this.height;
                        const nx = (x + dx + this.width) % this.width;
                        count += this.world[ny][nx];
                    }
                }
                
                // Apply rules
                if (this.world[y][x] === 1) {
                    newWorld[y][x] = this.getNthBit(this.sNum, count);
                } else {
                    newWorld[y][x] = this.getNthBit(this.bNum, count);
                }
            }
        }
        
        this.world = newWorld;
    }

    changeRule(sRule, bRule) {
        this.sNum = this.getRuleNumber(sRule);
        this.bNum = this.getRuleNumber(bRule);
        this.reset();
    }
}

const CA2D = ({ width = 400, height = 400, cellSize = 2 }) => {
    const canvasRef = useRef(null);
    const automaton = useRef(null);
    const animationFrame = useRef(null);
    const [isRunning, setIsRunning] = useState(true);
    const [isRandom, setIsRandom] = useState(false);

    const handleToggleRandom = () => {
        setIsRandom(!isRandom);
        // Immediately reinitialize the automaton with new random setting
        const gridWidth = Math.floor(width / cellSize);
        const gridHeight = Math.floor(height / cellSize);
        automaton.current = new CA2DAutomaton(gridWidth, gridHeight, "23", "3", !isRandom);
    };

    useEffect(() => {
        const gridWidth = Math.floor(width / cellSize);
        const gridHeight = Math.floor(height / cellSize);
        automaton.current = new CA2DAutomaton(gridWidth, gridHeight, "23", "3", isRandom);
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        const animate = () => {
            if (isRunning) {
                automaton.current.step();
                
                // Draw the current state
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, width, height);
                
                ctx.fillStyle = '#FFFFFF';
                for (let y = 0; y < automaton.current.height; y++) {
                    for (let x = 0; x < automaton.current.width; x++) {
                        if (automaton.current.world[y][x] === 1) {
                            ctx.fillRect(
                                x * cellSize, 
                                y * cellSize, 
                                cellSize, 
                                cellSize
                            );
                        }
                    }
                }
            }
            animationFrame.current = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            if (animationFrame.current) {
                cancelAnimationFrame(animationFrame.current);
            }
        };
    }, [width, height, cellSize, isRunning, isRandom]);

    const handleKeyDown = (e) => {
        switch (e.key) {
            case 'Delete':
            case 'Backspace':
                automaton.current.reset();
                break;
            case 'i':
                handleToggleRandom();
                break;
            case 'n':
                // Random rule
                const randomRule = () => {
                    const num = Math.floor(Math.random() * Math.pow(2, 9));
                    return automaton.current.getRuleString(num);
                };
                automaton.current.changeRule(randomRule(), randomRule());
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
                <button onClick={handleToggleRandom}>
                    Toggle Random Init
                </button>
            </div>
        </div>
    );
};

export default CA2D;
