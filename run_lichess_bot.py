import os
import sys
import argparse
from chess_bot.rl_agent import RLAgent
from chess_bot.adapter_lichess import LichessAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True, help='Lichess bot token with bot:play scope')
    parser.add_argument('--model', default='rl_model_selfplay.pt', help='RL model path')
    args = parser.parse_args()

    agent = RLAgent()
    try:
        agent.load(args.model)
        print(f'Loaded model {args.model}')
    except Exception as e:
        print('Failed to load model, starting with untrained agent:', e)

    adapter = LichessAdapter(args.token, agent)
    try:
        adapter.run()
    except KeyboardInterrupt:
        adapter.stop()
        print('Adapter stopped')


if __name__ == '__main__':
    main()
