# DDQN Traffic Light Control System

Deep Reinforcement Learning for Adaptive Traffic Signal Control using Double Deep Q-Network (DDQN) and SUMO traffic simulator.

## 🎯 Project Overview

This project implements an intelligent traffic light controller using Double Deep Q-Network (DDQN) reinforcement learning algorithm. The agent learns to optimize traffic flow at a 4-way intersection by minimizing vehicle waiting times and queue lengths.

**Key Features:**
- ✅ DDQN implementation with experience replay and target network
- ✅ SUMO traffic simulator integration via TraCI
- ✅ GPU-accelerated training (CUDA support)
- ✅ Real-time visualization with SUMO GUI
- ✅ Baseline comparisons (Fixed-time, Random policy)
- ✅ Comprehensive evaluation metrics

## 📊 Results

### Training Performance (500 Episodes)

| Metric | Initial (Ep 1-50) | Final (Ep 451-500) | Improvement |
|--------|-------------------|-------------------|-------------|
| **Average Reward** | -17,308 | **-6,670** | **+61%** |
| **Waiting Time** | 0.73s | 0.38s | -48% |
| **Queue Length** | 0.75 vehicles | 0.50 vehicles | -33% |

**Training Time:** ~75 minutes on NVIDIA RTX 4060 (GPU)

### Evaluation Results

[Add your evaluation results here after running evaluation]

**DDQN Agent Performance:**
- Average Waiting Time: [X.XX]s
- Average Queue Length: [X.XX] vehicles
- Total Reward: [XXXX]

**Comparison vs Baselines:**
- Fixed-Time Controller: [XX%] improvement
- Random Policy: [XX%] improvement

![Training Curves](results/training_curves.png)
![Comparison Plot](results/comparison_plot.png)

## 🛠️ Technologies Used

- **Python 3.13**
- **PyTorch 2.6.0** (CUDA 12.4 enabled)
- **SUMO 1.25.0** (Traffic Simulator)
- **NumPy, Matplotlib, Pandas, tqdm**

## 📁 Project Structure

```
.
├── agent.py                    # DDQN agent implementation
├── network.py                  # Neural network architecture
├── replay_buffer.py            # Experience replay buffer
├── sumo_environment.py         # SUMO environment wrapper
├── train.py                    # Training loop
├── evaluate.py                 # Evaluation functions
├── main.py                     # Entry point
├── generate_sumo_files.py      # SUMO configuration generator
├── experiment_manager.py       # Experiment tracking system
├── requirements.txt            # Python dependencies
├── checkpoints/                # Model checkpoints (every 100 episodes)
├── models/                     # Trained models
├── results/                    # Training history, plots
├── sumo_files/                 # SUMO network and route files
├── experiments/                # Saved experiment runs
└── IMPROVEMENT_GUIDE.md        # Performance tuning guide

```

## 🚀 Installation

### Prerequisites

1. **Python 3.10+**
2. **SUMO Traffic Simulator**
   - Download from: https://eclipse.dev/sumo/
   - Add to PATH: `C:\Program Files (x86)\Eclipse\Sumo\bin`

3. **CUDA Toolkit** (optional, for GPU acceleration)
   - CUDA 12.4+ recommended
   - NVIDIA GPU with compute capability 3.7+

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/DDQN-Traffic-Control.git
cd DDQN-Traffic-Control

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify SUMO installation
sumo --version
```

## 💻 Usage

### Training

Train a new DDQN agent for 500 episodes:

```bash
# Windows (with GPU cache workaround)
$env:PYTHONDONTWRITEBYTECODE="1"; python main.py --mode train --episodes 500

# Linux/Mac
PYTHONDONTWRITEBYTECODE=1 python main.py --mode train --episodes 500
```

**Training Options:**
```bash
python main.py --mode train --episodes 1000  # Extended training
python main.py --mode train --episodes 500 --seed 42  # Reproducible training
```

**Checkpoints saved at:** `checkpoints/ddqn_episode_100.pth`, `200.pth`, `300.pth`, `400.pth`, `500.pth`

### Evaluation

Evaluate trained agent with visualization:

```bash
# With SUMO GUI (visual simulation)
python main.py --mode evaluate --eval-episodes 10 --gui

# Without GUI (faster)
python main.py --mode evaluate --eval-episodes 100

# Test specific checkpoint
python main.py --mode evaluate --model-path checkpoints/ddqn_episode_400.pth --eval-episodes 50 --gui
```

### Full Pipeline

Train and evaluate in one command:

```bash
python main.py --mode all --episodes 500 --eval-episodes 100
```

## 🧪 Experiment Management

Track multiple training runs and compare results:

```python
from experiment_manager import save_current_training, ExperimentManager

# Save current training
save_current_training(
    name='baseline_500ep',
    description='Initial training with default hyperparameters',
    config={'episodes': 500, 'learning_rate': 0.001}
)

# List all experiments
manager = ExperimentManager()
manager.list_experiments()

# Compare experiments
manager.compare_experiments(['experiment_1_timestamp', 'experiment_2_timestamp'])

# Find best model
best = manager.get_best_experiment(metric='avg_reward')
```

## 🔧 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STATE_DIM` | 6 | Queue lengths (4) + phase + time |
| `ACTION_DIM` | 2 | Keep phase / Switch phase |
| `HIDDEN_DIM` | 128 | Neural network hidden units |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `GAMMA` | 0.95 | Discount factor |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_DECAY` | 0.995 | Epsilon decay per episode |
| `EPSILON_MIN` | 0.01 | Minimum exploration rate |
| `BATCH_SIZE` | 64 | Training batch size |
| `BUFFER_CAPACITY` | 10,000 | Replay buffer size |
| `TARGET_UPDATE_FREQ` | 10 | Episodes between target network updates |

See [IMPROVEMENT_GUIDE.md](IMPROVEMENT_GUIDE.md) for tuning suggestions.

## 📈 Performance Improvements

Want better results? See [IMPROVEMENT_GUIDE.md](IMPROVEMENT_GUIDE.md) for:
- Reward function tuning
- State representation enhancements
- Network architecture modifications
- Advanced RL techniques (Prioritized Replay, Dueling DQN, etc.)

## 🎮 SUMO GUI Controls

When running with `--gui` flag:
- **Space:** Pause/Resume
- **Arrow Keys:** Speed up/slow down simulation
- **Click vehicles:** View vehicle details
- **Right-click:** Zoom/pan view

## 📊 Files Included in Repo

### Always Included ✅
- ✅ All Python source code (`.py` files)
- ✅ Results: `results/training_history.csv`, `results/*.png`
- ✅ SUMO configuration: `sumo_files/*.xml`, `sumo_files/*.sumocfg`
- ✅ Documentation: `README.md`, `IMPROVEMENT_GUIDE.md`
- ✅ Dependencies: `requirements.txt`

### Model Files (Conditional)
- ✅ **Trained models** (`models/*.pth`, `checkpoints/*.pth`) - Included if <100MB each
- ⚠️ If models are >100MB, use [Git LFS](https://git-lfs.github.com/) or host separately
- 💡 Provide download link in README if models are too large

### Excluded ❌
- ❌ Python cache (`__pycache__/`, `*.pyc`)
- ❌ Virtual environment (`venv/`, `.venv/`)
- ❌ IDE files (`.vscode/`, `.idea/`)
- ❌ OS files (`.DS_Store`, `Thumbs.db`)

## 🤝 Contributing

Improvements welcome! Areas for contribution:
- [ ] Prioritized Experience Replay
- [ ] Dueling DQN architecture
- [ ] Multi-intersection coordination
- [ ] Real-world traffic pattern data
- [ ] Hyperparameter optimization (Optuna)

## 📝 License

MIT License - See LICENSE file for details

## 👨‍💻 Author

[Your Name]
- GitHub: [@YourUsername](https://github.com/YourUsername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- SUMO Traffic Simulator: [https://eclipse.dev/sumo/](https://eclipse.dev/sumo/)
- Deep RL Course: [Your course/university if applicable]
- Inspired by: [List any papers or projects]

## 📚 References

1. van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI.
2. SUMO Documentation: https://sumo.dlr.de/docs/
3. PyTorch Documentation: https://pytorch.org/docs/

---

**Note:** This is a semester project demonstrating reinforcement learning for traffic control. For real-world deployment, additional validation and safety measures would be required.
