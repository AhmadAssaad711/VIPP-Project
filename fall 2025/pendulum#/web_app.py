"""
Web-based CartPole RL Visualization
Interactive web interface to choose and visualize different RL methods
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import gymnasium as gym
import numpy as np
import json
import asyncio
from q_learning import QLearningAgent
from policy_gradient_simple import PolicyGradientAgent
from model_based import LQRController

app = FastAPI()

# HTML Frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>CartPole RL Visualizer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .content {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 0;
        }
        .controls {
            background: #f8f9fa;
            padding: 30px;
            border-right: 2px solid #e0e0e0;
        }
        .control-group {
            margin-bottom: 25px;
        }
        .control-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
            font-size: 1.1em;
        }
        .method-selector {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .method-btn {
            padding: 15px;
            border: 2px solid #ddd;
            background: white;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 1em;
            font-weight: 500;
        }
        .method-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .method-btn.active {
            border-color: #667eea;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .method-btn.pg { border-left: 4px solid #9c27b0; }
        .method-btn.q { border-left: 4px solid #2196f3; }
        .method-btn.lqr { border-left: 4px solid #f44336; }
        
        input[type="range"] {
            width: 100%;
            margin-top: 5px;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
        }
        .start-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .start-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        .start-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .stop-btn {
            width: 100%;
            padding: 15px;
            background: #f44336;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
        }
        .visualization {
            padding: 30px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        #canvas {
            border: 3px solid #667eea;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            background: white;
        }
        .stats {
            width: 100%;
            max-width: 800px;
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: 700;
            color: #333;
        }
        .method-info {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 0.9em;
            line-height: 1.6;
        }
        .status {
            padding: 10px;
            background: #fff3cd;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: 500;
        }
        .status.running {
            background: #d1ecf1;
            color: #0c5460;
        }
        .status.complete {
            background: #d4edda;
            color: #155724;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 CartPole RL Visualizer</h1>
            <p>Interactive visualization of reinforcement learning methods</p>
        </div>
        
        <div class="content">
            <div class="controls">
                <div class="control-group">
                    <label>Select RL Method</label>
                    <div class="method-selector">
                        <button class="method-btn pg" onclick="selectMethod('policy_gradient')">
                            📊 Policy Gradient<br>
                            <small style="opacity:0.8">REINFORCE Algorithm</small>
                        </button>
                        <button class="method-btn q" onclick="selectMethod('q_learning')">
                            🎯 Q-Learning<br>
                            <small style="opacity:0.8">Value-Based Method</small>
                        </button>
                        <button class="method-btn lqr" onclick="selectMethod('lqr')">
                            🔬 Model-Based LQR<br>
                            <small style="opacity:0.8">Optimal Control</small>
                        </button>
                    </div>
                </div>
                
                <div class="control-group">
                    <label>Number of Episodes</label>
                    <input type="number" id="episodes" value="100" min="1" max="500">
                </div>
                
                <div class="control-group">
                    <label>Animation Speed</label>
                    <input type="range" id="speed" min="1" max="100" value="50">
                    <small>Slower ← → Faster</small>
                </div>
                
                <button class="start-btn" onclick="startTraining()">Start Training</button>
                <button class="stop-btn" onclick="stopTraining()" style="display:none;">Stop</button>
                
                <div class="method-info" id="methodInfo">
                    <strong>Select a method above to see details</strong>
                </div>
            </div>
            
            <div class="visualization">
                <div class="status" id="status">Ready to start</div>
                
                <canvas id="canvas" width="800" height="500"></canvas>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-label">Episode</div>
                        <div class="stat-value" id="episode">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Steps</div>
                        <div class="stat-value" id="steps">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Current Reward</div>
                        <div class="stat-value" id="reward">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Average (10)</div>
                        <div class="stat-value" id="average">0</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let selectedMethod = null;
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        const methodInfo = {
            'policy_gradient': {
                name: 'Policy Gradient (REINFORCE)',
                color: '#9c27b0',
                info: '<strong>Policy Gradient</strong><br>Learns a policy directly by gradient ascent. Updates policy parameters based on episode returns. Expect gradual improvement over 100+ episodes.'
            },
            'q_learning': {
                name: 'Q-Learning',
                color: '#2196f3',
                info: '<strong>Q-Learning</strong><br>Value-based method that learns Q(s,a) function. Uses epsilon-greedy exploration. Learns slowly but steadily over 150+ episodes.'
            },
            'lqr': {
                name: 'Model-Based LQR',
                color: '#f44336',
                info: '<strong>Linear Quadratic Regulator</strong><br>Uses known dynamics to compute optimal control. Perfect 500-step performance from episode 1. No learning needed!'
            }
        };
        
        function selectMethod(method) {
            selectedMethod = method;
            document.querySelectorAll('.method-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.closest('.method-btn').classList.add('active');
            
            document.getElementById('methodInfo').innerHTML = methodInfo[method].info;
        }
        
        function startTraining() {
            if (!selectedMethod) {
                alert('Please select a method first!');
                return;
            }
            
            const episodes = parseInt(document.getElementById('episodes').value);
            const speed = parseInt(document.getElementById('speed').value);
            
            document.querySelector('.start-btn').style.display = 'none';
            document.querySelector('.stop-btn').style.display = 'block';
            document.getElementById('status').textContent = 'Connecting...';
            document.getElementById('status').className = 'status running';
            
            ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                ws.send(JSON.stringify({
                    method: selectedMethod,
                    episodes: episodes,
                    speed: speed
                }));
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'state') {
                    drawCartPole(data);
                    updateStats(data);
                } else if (data.type === 'complete') {
                    document.getElementById('status').textContent = 'Training Complete!';
                    document.getElementById('status').className = 'status complete';
                    document.querySelector('.start-btn').style.display = 'block';
                    document.querySelector('.stop-btn').style.display = 'none';
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                document.getElementById('status').textContent = 'Connection error!';
            };
        }
        
        function stopTraining() {
            if (ws) {
                ws.close();
            }
            document.querySelector('.start-btn').style.display = 'block';
            document.querySelector('.stop-btn').style.display = 'none';
            document.getElementById('status').textContent = 'Stopped';
            document.getElementById('status').className = 'status';
        }
        
        function drawCartPole(data) {
            const { state, method } = data;
            const [x_pos, x_vel, theta, theta_vel] = state;
            
            // Clear canvas
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw track
            ctx.strokeStyle = '#cccccc';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(50, 350);
            ctx.lineTo(750, 350);
            ctx.stroke();
            
            // Cart position
            const cartX = 400 + x_pos * 100;
            const cartY = 350;
            const cartWidth = 70;
            const cartHeight = 35;
            
            // Draw cart
            const color = methodInfo[method].color;
            ctx.fillStyle = color;
            ctx.fillRect(cartX - cartWidth/2, cartY - cartHeight/2, cartWidth, cartHeight);
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 3;
            ctx.strokeRect(cartX - cartWidth/2, cartY - cartHeight/2, cartWidth, cartHeight);
            
            // Draw pole
            const poleLength = 140;
            const poleEndX = cartX + poleLength * Math.sin(theta);
            const poleEndY = cartY - poleLength * Math.cos(theta);
            
            const angleDeg = Math.abs(theta * 180 / Math.PI);
            let poleColor = '#52a83a';
            if (angleDeg > 15) poleColor = '#f44336';
            else if (angleDeg > 8) poleColor = '#9c27b0';
            
            ctx.strokeStyle = poleColor;
            ctx.lineWidth = 10;
            ctx.beginPath();
            ctx.moveTo(cartX, cartY);
            ctx.lineTo(poleEndX, poleEndY);
            ctx.stroke();
            
            // Draw joints
            ctx.fillStyle = '#000000';
            ctx.beginPath();
            ctx.arc(cartX, cartY, 8, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.fillStyle = poleColor;
            ctx.beginPath();
            ctx.arc(poleEndX, poleEndY, 10, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw angle indicator
            ctx.font = '16px Arial';
            ctx.fillStyle = '#333';
            ctx.fillText(`Angle: ${angleDeg.toFixed(1)}°`, 20, 30);
            ctx.fillText(`Position: ${x_pos.toFixed(2)}`, 20, 55);
        }
        
        function updateStats(data) {
            document.getElementById('episode').textContent = data.episode;
            document.getElementById('steps').textContent = data.step;
            document.getElementById('reward').textContent = data.reward;
            document.getElementById('average').textContent = data.average.toFixed(1);
            document.getElementById('status').textContent = 
                `Training ${methodInfo[data.method].name} - Episode ${data.episode}`;
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get_home():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Receive configuration
        config = await websocket.receive_json()
        method = config['method']
        episodes = config['episodes']
        speed = config['speed']
        
        # Create agent
        if method == 'policy_gradient':
            agent = PolicyGradientAgent(learning_rate=0.01, discount_factor=0.99)
        elif method == 'q_learning':
            agent = QLearningAgent()
        elif method == 'lqr':
            agent = LQRController()
        else:
            await websocket.close()
            return
        
        env = gym.make('CartPole-v1')
        rewards = []
        
        # Training loop
        for episode in range(episodes):
            state, _ = env.reset()
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()
            
            episode_reward = 0
            
            for step in range(500):
                # Get action
                if isinstance(agent, QLearningAgent):
                    action = agent.get_action(state, explore=(episode < episodes - 10))
                elif isinstance(agent, PolicyGradientAgent):
                    action = agent.get_action(state)
                elif isinstance(agent, LQRController):
                    action = agent.get_action(state)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # Update agent
                if isinstance(agent, QLearningAgent):
                    agent.update(state, action, reward, next_state, terminated or truncated)
                elif isinstance(agent, PolicyGradientAgent):
                    agent.store_reward(reward)
                
                state = next_state
                episode_reward += reward
                
                # Send state update (throttled by speed)
                if step % max(1, 10 - speed // 10) == 0:
                    avg = np.mean(rewards[-10:]) if rewards else 0
                    await websocket.send_json({
                        'type': 'state',
                        'state': state.tolist(),
                        'episode': episode + 1,
                        'step': step,
                        'reward': int(episode_reward),
                        'average': float(avg),
                        'method': method
                    })
                    await asyncio.sleep(0.01)
                
                if terminated or truncated:
                    break
            
            # Update policy for PG
            if isinstance(agent, PolicyGradientAgent):
                agent.update()
            
            rewards.append(episode_reward)
        
        env.close()
        
        # Send completion message
        await websocket.send_json({
            'type': 'complete',
            'final_average': float(np.mean(rewards[-10:])),
            'best': int(max(rewards))
        })
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("CartPole RL Web Visualizer")
    print("="*70)
    print("\nStarting server at: http://localhost:8000")
    print("Open this URL in your browser to use the interactive visualizer!")
    print("\nPress Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
