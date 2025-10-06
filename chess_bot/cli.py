import argparse
import chess
from chess_bot.engine import SimpleEngine

try:
    from chess_bot.rl_agent import RLAgent
except Exception:
    RLAgent = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-model", help="path to RL model to load (optional)")
    args = parser.parse_args()

    board = chess.Board()
    engine = SimpleEngine(depth=2)
    rl_agent = None
    if args.rl_model and RLAgent is not None:
        rl_agent = RLAgent()
        try:
            rl_agent.load(args.rl_model)
            print(f"Loaded RL model from {args.rl_model}")
        except Exception as e:
            print(f"Failed to load RL model: {e}")

    print("Welcome to Chess Bot CLI. Enter moves in UCI format (e.g. e2e4). Type 'quit' to exit.")
    print(board)
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            user = input("Your move: ").strip()
            if user.lower() in ("quit", "exit"):
                print("Goodbye")
                return
            try:
                move = chess.Move.from_uci(user)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
                    continue
            except Exception:
                print("Invalid UCI. Try again.")
                continue
        else:
            print("Engine thinking...")
            if rl_agent is not None:
                move = rl_agent.select_move(board)
            else:
                move = engine.search(board)
            if move is None:
                print("No move found")
                break
            board.push(move)
            print(f"Engine plays: {move}")
        print(board)
    print("Game over:", board.result())


if __name__ == "__main__":
    main()
