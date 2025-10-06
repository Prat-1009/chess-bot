# Chess Bot

A minimal Python chess bot using python-chess with a simple CLI.

Requirements

- Python 3.8+
- see `requirements.txt`

Quick start

Create a virtualenv and install requirements, then run the CLI:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
python -m chess_bot.cli
```

If you use the provided workspace conda environment, run the CLI with the following pattern:

```powershell
D:/numpy/Scripts/conda.exe run -p "d:\\projects\\aiml projects\\chess bot\\.conda" --no-capture-output python -m chess_bot.cli
```

Usage

- Play against the bot in UCI move format (e.g. `e2e4`).
- The bot will respond with its move.

Files

- `chess_bot/engine.py` - engine with minimax + alpha-beta
- `chess_bot/cli.py` - simple CLI
- `tests/test_engine.py` - minimal tests

Example session

```powershell
# start the CLI (using the workspace conda env if configured)
# then enter moves like:
# Your move: e2e4
# Engine plays: e7e5
# Your move: g1f3
```

Reinforcement learning agent

You can train a tiny REINFORCE agent (PyTorch) with:

```powershell
D:/numpy/Scripts/conda.exe run -p "d:\\projects\\aiml projects\\chess bot\\.conda" --no-capture-output python train_rl.py
```

That will create `rl_model.pt` in the workspace. To run the CLI using the saved model:

```powershell
python -m chess_bot.cli --rl-model rl_model.pt
```

Self-play training

The trainer now uses self-play (the agent plays both sides). You can specify the number of episodes and save path:

```powershell
D:/numpy/Scripts/conda.exe run -p "d:\\projects\\aiml projects\\chess bot\\.conda" --no-capture-output python train_rl.py --episodes 200 --save-path rl_model_selfplay.pt
```

Note: full training on CPU may take a long time. Consider using fewer episodes for a quick smoke test (e.g. `--episodes 20`) or running on a GPU-enabled environment.

Chess.com (Playwright automation) — experimental

This repository includes an experimental Playwright-based adapter `chess_bot/adapter_chesscom.py` and a runner `run_chesscom_bot.py`.

Important safety/legal note: Automating chess.com may violate their Terms of Service. Use only with accounts you own and preferably a throwaway account.

Install Playwright and browsers:

```powershell
pip install -r requirements.txt
python -m playwright install
```

Run the adapter (dry-run recommended first):

```powershell
python run_chesscom_bot.py --username yourname --game-url "https://www.chess.com/live/game/XXXXX" --dry-run
```

To actually apply moves (not recommended for production): remove `--dry-run` and ensure you understand the risks.


Playing on real platforms

Lichess
- This project includes a minimal Lichess adapter: `chess_bot/adapter_lichess.py`.
- To run as a Lichess bot you need a bot token (with `bot:play`) and then run a small runner that uses `LichessAdapter` with your agent instance.

chess.com
- chess.com does not provide a public bot API for automated play in the same way Lichess does. Automating play on chess.com (login automation, UI scraping, or using unofficial endpoints) likely violates chess.com's Terms of Service and may get your account banned.
- For chess.com integration you would need to use their documented APIs, or obtain explicit permission. I recommend using Lichess for automated/bot play.
