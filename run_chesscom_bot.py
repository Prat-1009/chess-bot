import argparse
from chess_bot.rl_agent import RLAgent
from chess_bot.adapter_chesscom import ChessComAdapter
import logging


def setup_logging(level: str):
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logging.basicConfig(level=numeric, format='%(asctime)s %(levelname)s %(name)s: %(message)s')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', required=True, help='chess.com username (for detection)')
    parser.add_argument('--password', required=False, help='password (optional if session stored)')
    parser.add_argument('--game-url', required=True, help='URL of the live game page to watch')
    parser.add_argument('--model', default='rl_model_selfplay.pt', help='RL model path')
    parser.add_argument('--log-level', default='INFO', help='logging level (DEBUG, INFO, WARNING)')
    parser.add_argument('--storage-state', default='playwright_storage.json', help='Path to save/load Playwright storage_state (cookies/localStorage)')
    parser.add_argument('--dry-run', action='store_true', help='If set, do not actually click to play')
    parser.add_argument('--headful', action='store_true', help='Launch browser in headful mode for debugging')
    parser.add_argument('--browser-channel', default=None, help='Playwright browser channel to use (e.g., "chrome", "msedge")')
    parser.add_argument('--show-suggestions', action='store_true', help='Show move suggestions overlay on the page')
    parser.add_argument('--auto-confirm', action='store_true', help='Auto-confirm suggestions and apply moves without manual confirmation')
    args = parser.parse_args()

    setup_logging(args.log_level)

    agent = RLAgent()
    try:
        agent.load(args.model)
        print(f'Loaded model {args.model}')
    except Exception as e:
        print('Failed to load model (continuing with untrained agent):', e)

    adapter = ChessComAdapter(agent, headless=not args.headful)
    adapter.run(args.username, args.password or '', args.game_url, dry_run=args.dry_run, headful=args.headful, browser_channel=args.browser_channel, show_suggestions=args.show_suggestions, auto_confirm=args.auto_confirm, storage_state=args.storage_state)


if __name__ == '__main__':
    main()
