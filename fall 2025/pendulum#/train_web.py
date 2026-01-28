"""
CartPole Q-Learning Web Server with WebSocket Streaming

This FastAPI server trains a Q-learning agent and streams
the training progress to a web browser in real-time via WebSocket.

The frontend (index.html) renders the CartPole on an HTML canvas
and displays training statistics as they update.

Run with: uvicorn train_web:app --reload
Then open: http://localhost:8000
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import gymnasium as gym
import numpy as np
import json
import asyncio
from q_agent import QLearningAgent
from policy_gradient_simple import PolicyGradientAgent
from model_based import LQRController
from pathlib import Path


app = FastAPI(title="CartPole Q-Learning Live Training")


# HTML template for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CartPole Q-Learning - Live Training</title>
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
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 30px;
            max-width: 1200px;
            width: 100%;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        
        .content {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
        }
        
        .canvas-container {
            background: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        canvas {
            border: 2px solid #ddd;
            border-radius: 8px;
            background: white;
        }
        
        .stats-panel {
            background: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .stat-group {
            background: white;
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #667eea;
        }
        
        .stat-group h3 {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 0.95em;
        }
        
        .stat-label {
            color: #666;
            font-weight: 500;
        }
        
        .stat-value {
            color: #333;
            font-weight: 700;
            font-family: 'Courier New', monospace;
        }
        
        .stat-value.good {
            color: #10b981;
        }
        
        .stat-value.warning {
            color: #f59e0b;
        }
        
        .stat-value.danger {
            color: #ef4444;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s ease-in-out infinite;
        }
        
        .status-dot.disconnected {
            background: #ef4444;
            animation: none;
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }
        
        .control-buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 0.9em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        button.primary {
            background: #667eea;
            color: white;
        }
        
        button.primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        button.secondary {
            background: #e5e7eb;
            color: #333;
        }
        
        button.secondary:hover {
            background: #d1d5db;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            width: 0%;
        }
        
        @media (max-width: 968px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 CartPole Q-Learning</h1>
        <p class="subtitle">Watch the AI learn to balance the pole in real-time</p>
        
        <div class="content">
            <div class="canvas-container">
                <canvas id="cartpole-canvas" width="700" height="500"></canvas>
            </div>
            
            <div class="stats-panel">
                <div class="status-indicator">
                    <div class="status-dot" id="status-dot"></div>
                    <span id="status-text">Connecting...</span>
                </div>
                
                <div class="stat-group">
                    <h3>Training Progress</h3>
                    <div class="stat-item">
                        <span class="stat-label">Episode:</span>
                        <span class="stat-value" id="episode">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Step:</span>
                        <span class="stat-value" id="step">0</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="episode-progress"></div>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h3>Performance</h3>
                    <div class="stat-item">
                        <span class="stat-label">Reward:</span>
                        <span class="stat-value" id="reward">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg (100):</span>
                        <span class="stat-value" id="avg-reward">0.0</span>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h3>Agent State</h3>
                    <div class="stat-item">
                        <span class="stat-label">Epsilon:</span>
                        <span class="stat-value" id="epsilon">1.000</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Q Mean:</span>
                        <span class="stat-value" id="q-mean">0.000</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Q Max:</span>
                        <span class="stat-value" id="q-max">0.000</span>
                    </div>
                </div>
                
                <div class="stat-group">
                    <h3>Environment State</h3>
                    <div class="stat-item">
                        <span class="stat-label">Cart Pos:</span>
                        <span class="stat-value" id="cart-pos">0.000</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Cart Vel:</span>
                        <span class="stat-value" id="cart-vel">0.000</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Pole Angle:</span>
                        <span class="stat-value" id="pole-angle">0.000</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Ang Vel:</span>
                        <span class="stat-value" id="ang-vel">0.000</span>
                    </div>
                </div>
                
                <div class="control-buttons">
                    <label style="display:flex;gap:8px;align-items:center;flex:1">
                        <select id="method-select" style="flex:1;padding:8px;border-radius:6px;border:1px solid #ddd;">
                            <option value="q">Q-Learning (Value-Based)</option>
                            <option value="pg">Policy Gradient (REINFORCE)</option>
                            <option value="lqr">Model-Based (LQR)</option>
                        </select>
                    </label>
                    <label style="display:flex;gap:8px;align-items:center;">
                        <input id="episodes-input" type="number" min="1" max="2000" value="500" style="width:90px;padding:8px;border-radius:6px;border:1px solid #ddd;"/>
                    </label>
                    <button class="primary" id="start-btn" onclick="startTraining()">Start</button>
                    <button class="secondary" id="stop-btn" onclick="stopTraining()" disabled>Stop</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('cartpole-canvas');
        const ctx = canvas.getContext('2d');
        
        let ws = null;
        let isTraining = false;
        
        // Canvas dimensions
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        
        // Cart and pole dimensions
        const cartWidth = 80;
        const cartHeight = 40;
        const poleLength = 150;
        const poleWidth = 8;
        const trackY = canvasHeight * 0.6;
        const trackLeft = 50;
        const trackRight = canvasWidth - 50;
        
        function drawCartPole(state) {
            // Clear canvas
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvasWidth, canvasHeight);
            
            const [cartPos, cartVel, poleAngle, angVel] = state;
            
            // Draw track
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(trackLeft, trackY);
            ctx.lineTo(trackRight, trackY);
            ctx.stroke();
            
            // Map cart position to canvas
            const cartX = trackLeft + (cartPos + 4.8) / 9.6 * (trackRight - trackLeft);
            const cartY = trackY;
            
            // Draw cart
            ctx.fillStyle = '#3b82f6';
            ctx.fillRect(cartX - cartWidth/2, cartY - cartHeight, cartWidth, cartHeight);
            ctx.strokeStyle = '#1e40af';
            ctx.lineWidth = 2;
            ctx.strokeRect(cartX - cartWidth/2, cartY - cartHeight, cartWidth, cartHeight);
            
            // Draw wheels
            ctx.fillStyle = '#1f2937';
            ctx.beginPath();
            ctx.arc(cartX - 20, cartY + 2, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.arc(cartX + 20, cartY + 2, 8, 0, Math.PI * 2);
            ctx.fill();
            
            // Calculate pole end position
            const poleEndX = cartX + poleLength * Math.sin(poleAngle);
            const poleEndY = cartY - cartHeight - poleLength * Math.cos(poleAngle);
            
            // Color pole based on angle
            const angleSeverity = Math.abs(poleAngle) / 0.418;
            const red = Math.floor(255 * angleSeverity + 50 * (1 - angleSeverity));
            const green = Math.floor(50 * angleSeverity + 255 * (1 - angleSeverity));
            const blue = 50;
            const poleColor = `rgb(${red}, ${green}, ${blue})`;
            
            // Draw pole
            ctx.strokeStyle = poleColor;
            ctx.lineWidth = poleWidth;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(cartX, cartY - cartHeight);
            ctx.lineTo(poleEndX, poleEndY);
            ctx.stroke();
            
            // Draw joint
            ctx.fillStyle = '#1f2937';
            ctx.beginPath();
            ctx.arc(cartX, cartY - cartHeight, 6, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw pole tip
            ctx.fillStyle = poleColor;
            ctx.beginPath();
            ctx.arc(poleEndX, poleEndY, 10, 0, Math.PI * 2);
            ctx.fill();
        }
        
        function updateStats(data) {
            document.getElementById('episode').textContent = data.episode;
            document.getElementById('step').textContent = data.step;
            document.getElementById('reward').textContent = data.total_reward.toFixed(0);
            
            const avgReward = data.avg_reward.toFixed(1);
            const avgRewardElem = document.getElementById('avg-reward');
            avgRewardElem.textContent = avgReward;
            avgRewardElem.className = 'stat-value ' + 
                (data.avg_reward > 150 ? 'good' : data.avg_reward > 50 ? 'warning' : 'danger');
            
            document.getElementById('epsilon').textContent = data.epsilon.toFixed(3);
            document.getElementById('q-mean').textContent = data.q_stats.q_mean.toFixed(3);
            document.getElementById('q-max').textContent = data.q_stats.q_max.toFixed(3);
            
            document.getElementById('cart-pos').textContent = data.state[0].toFixed(3);
            document.getElementById('cart-vel').textContent = data.state[1].toFixed(3);
            
            const angleElem = document.getElementById('pole-angle');
            angleElem.textContent = data.state[2].toFixed(3);
            angleElem.className = 'stat-value ' +
                (Math.abs(data.state[2]) > 0.3 ? 'danger' : Math.abs(data.state[2]) > 0.15 ? 'warning' : 'good');
            
            document.getElementById('ang-vel').textContent = data.state[3].toFixed(3);
            
            // Update progress bar
            const progress = (data.step / 500) * 100;
            document.getElementById('episode-progress').style.width = progress + '%';
        }
        
        function startTraining() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.close();
            }

            const method = document.getElementById('method-select').value;
            const episodes = Number(document.getElementById('episodes-input').value) || 500;
            const url = `ws://localhost:8000/ws/train?method=${encodeURIComponent(method)}&episodes=${encodeURIComponent(episodes)}`;
            ws = new WebSocket(url);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('status-text').textContent = 'Training...';
                document.getElementById('status-dot').classList.remove('disconnected');
                document.getElementById('start-btn').disabled = true;
                document.getElementById('stop-btn').disabled = false;
                isTraining = true;
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'step') {
                    drawCartPole(data.state);
                    updateStats(data);
                } else if (data.type === 'complete') {
                    document.getElementById('status-text').textContent = 
                        `Training Complete! Final Avg: ${data.final_avg_reward.toFixed(1)}`;
                    isTraining = false;
                    document.getElementById('start-btn').disabled = false;
                    document.getElementById('stop-btn').disabled = true;
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                document.getElementById('status-text').textContent = 'Error connecting';
                document.getElementById('status-dot').classList.add('disconnected');
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                if (isTraining) {
                    document.getElementById('status-text').textContent = 'Stopped';
                }
                document.getElementById('status-dot').classList.add('disconnected');
                document.getElementById('start-btn').disabled = false;
                document.getElementById('stop-btn').disabled = true;
                isTraining = false;
            };
        }
        
        function stopTraining() {
            if (ws) {
                ws.close();
            }
        }
        
        // Initial draw
        drawCartPole([0, 0, 0, 0]);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Serve the main HTML page."""
    return HTML_TEMPLATE


@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    """WebSocket endpoint that streams training/evaluation for different agents.

    Query params (on WebSocket URL):
      - method: 'q' | 'pg' | 'lqr' (default 'q')
      - episodes: integer number of episodes to run (default 500)
    """
    await websocket.accept()

    # Parse query params from the WebSocket connection
    method = websocket.query_params.get('method', 'q')
    try:
        num_episodes = int(websocket.query_params.get('episodes', 500))
    except Exception:
        num_episodes = 500

    max_steps = 500
    env = gym.make('CartPole-v1')

    print(f"WebSocket client connected. method={method} episodes={num_episodes}")

    try:
        if method == 'q':
            # Q-Learning (discretized)
            agent = QLearningAgent(
                n_bins=(6, 6, 12, 12),
                learning_rate=0.1,
                discount_factor=0.99,
                epsilon_start=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995
            )

            episode_rewards = []
            for episode in range(num_episodes):
                state, _ = env.reset()
                discrete_state = agent.discretize_state(state)
                total_reward = 0.0

                for step in range(max_steps):
                    action = agent.get_action(discrete_state, explore=True)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    next_discrete_state = agent.discretize_state(next_state)
                    agent.update(discrete_state, action, reward, next_discrete_state, done)

                    total_reward += reward

                    avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0

                    update_data = {
                        'type': 'step',
                        'episode': episode + 1,
                        'step': step + 1,
                        'state': next_state.tolist(),
                        'total_reward': float(total_reward),
                        'avg_reward': avg_reward,
                        'epsilon': float(agent.epsilon),
                        'q_stats': {
                            'q_mean': float(agent.get_stats()['q_mean']),
                            'q_max': float(agent.get_stats()['q_max']),
                            'q_min': float(agent.get_stats()['q_min'])
                        }
                    }

                    await websocket.send_json(update_data)
                    await asyncio.sleep(0.01)

                    state = next_state
                    discrete_state = next_discrete_state

                    if done:
                        break

                episode_rewards.append(total_reward)
                agent.decay_epsilon()

                if (episode + 1) % 10 == 0:
                    avg = np.mean(episode_rewards[-100:])
                    print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.0f} | Avg(100): {avg:.1f} | Epsilon: {agent.epsilon:.3f}")

            final_avg = float(np.mean(episode_rewards[-100:]))
            await websocket.send_json({'type': 'complete', 'final_avg_reward': final_avg, 'total_episodes': num_episodes})

        elif method == 'pg':
            # Policy Gradient (REINFORCE) using NumPy implementation
            agent = PolicyGradientAgent(learning_rate=0.005, discount_factor=0.99)
            episode_rewards = []

            for episode in range(num_episodes):
                state, _ = env.reset()
                agent.reset_episode()
                total_reward = 0.0

                for step in range(max_steps):
                    action = agent.get_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    agent.store_reward(reward)
                    total_reward += reward

                    avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0

                    update_data = {
                        'type': 'step',
                        'episode': episode + 1,
                        'step': step + 1,
                        'state': next_state.tolist(),
                        'total_reward': float(total_reward),
                        'avg_reward': avg_reward,
                        'epsilon': 0.0,
                        'q_stats': {'q_mean': 0.0, 'q_max': 0.0, 'q_min': 0.0}
                    }

                    await websocket.send_json(update_data)
                    await asyncio.sleep(0.01)

                    state = next_state

                    if done:
                        break

                loss = agent.update()
                episode_rewards.append(total_reward)

                if (episode + 1) % 10 == 0:
                    avg = np.mean(episode_rewards[-100:])
                    print(f"PG Episode {episode + 1}/{num_episodes} | Reward: {total_reward:.0f} | Avg(100): {avg:.1f}")

            final_avg = float(np.mean(episode_rewards[-100:]))
            await websocket.send_json({'type': 'complete', 'final_avg_reward': final_avg, 'total_episodes': num_episodes})

        elif method == 'lqr':
            # Model-based LQR evaluation
            controller = LQRController()
            episode_rewards = []

            for episode in range(num_episodes):
                state, _ = env.reset()
                total_reward = 0.0

                for step in range(max_steps):
                    action = controller.get_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    total_reward += reward

                    avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0

                    update_data = {
                        'type': 'step',
                        'episode': episode + 1,
                        'step': step + 1,
                        'state': next_state.tolist(),
                        'total_reward': float(total_reward),
                        'avg_reward': avg_reward,
                        'epsilon': 0.0,
                        'q_stats': {'q_mean': 0.0, 'q_max': 0.0, 'q_min': 0.0}
                    }

                    await websocket.send_json(update_data)
                    await asyncio.sleep(0.005)

                    state = next_state

                    if done:
                        break

                episode_rewards.append(total_reward)

            final_avg = float(np.mean(episode_rewards[-100:]))
            await websocket.send_json({'type': 'complete', 'final_avg_reward': final_avg, 'total_episodes': num_episodes})

        else:
            await websocket.send_json({'type': 'error', 'message': f'Unknown method: {method}'})

    except WebSocketDisconnect:
        print("Client disconnected during training")
    except Exception as e:
        print(f"Error during training: {e}")
        try:
            await websocket.send_json({'type': 'error', 'message': str(e)})
        except Exception:
            pass
    finally:
        env.close()


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting CartPole Q-Learning Web Server")
    print("="*60)
    print("\nOpen your browser and navigate to:")
    print("  → http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
