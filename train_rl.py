import argparse
from chess_bot.rl_agent import RLAgent, Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200, help="number of episodes to train")
    parser.add_argument("--save-path", default="rl_model.pt", help="where to save the trained model")
    args = parser.parse_args()

    agent = RLAgent()
    trainer = Trainer(agent)
    trainer.train(episodes=args.episodes, verbose=True)
    agent.save(args.save_path)
    print(f'Saved model to {args.save_path}')


if __name__ == '__main__':
    main()
