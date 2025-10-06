"""
Minimal Lichess Bot adapter.

How it works:
- Uses Lichess Bot API (SSE event stream at /api/stream/event)
- When a game starts, streams the game's state at /api/board/game/stream/{gameId}
- Uses the provided RLAgent (or engine) to pick moves and posts them to Lichess

Important: you must provide a Lichess API token with 'bot:play' scope. Do NOT share your token.
This example is minimal and intended as a starting point. Respect Lichess TOS and rate limits.
"""

import requests
import json
import threading
import time
import chess
from typing import Optional


class LichessAdapter:
    def __init__(self, token: str, agent, base_url: str = "https://lichess.org"):
        """token: personal bot token with bot:play scope; agent: object exposing select_move(board) -> chess.Move"""
        self.token = token
        self.agent = agent
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self._stop = False
        self.account_id = None

    def get_account_id(self) -> Optional[str]:
        if self.account_id:
            return self.account_id
        r = self.session.get(f"{self.base}/api/account")
        if r.status_code == 200:
            data = r.json()
            self.account_id = data.get("id")
            return self.account_id
        return None

    def stream_events(self):
        url = f"{self.base}/api/stream/event"
        with self.session.get(url, stream=True) as resp:
            for raw in resp.iter_lines(decode_unicode=True):
                if self._stop:
                    break
                if not raw:
                    continue
                line = raw.strip()
                # SSE data lines start with 'data: '
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if not payload or payload == "\n":
                        continue
                    try:
                        obj = json.loads(payload)
                    except Exception:
                        continue
                    yield obj

    def send_move(self, game_id: str, move_uci: str) -> bool:
        url = f"{self.base}/api/board/game/{game_id}/move/{move_uci}"
        r = self.session.post(url)
        return r.status_code in (200, 201)

    def handle_game(self, game_id: str):
        # Stream a single game's state and react when it's our turn
        url = f"{self.base}/api/board/game/stream/{game_id}"
        with self.session.get(url, stream=True) as resp:
            buffer = ""
            for raw in resp.iter_lines(decode_unicode=True):
                if self._stop:
                    break
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    try:
                        ev = json.loads(payload)
                    except Exception:
                        continue
                    # initial full game contains players and state
                    if ev.get("type") in ("gameFull", "gameState"):
                        # Reconstruct board from moves
                        moves_str = ev.get("state", {}).get("moves") or ev.get("moves")
                        board = chess.Board()
                        if moves_str:
                            for mv in moves_str.split():
                                try:
                                    board.push_uci(mv)
                                except Exception:
                                    # ignore illegal parse
                                    pass
                        # Determine our color: check players in gameFull
                        our_id = self.get_account_id()
                        white_id = None
                        black_id = None
                        if ev.get("white"):
                            white_id = ev["white"].get("user", {}).get("name") or ev["white"].get("id")
                        if ev.get("black"):
                            black_id = ev["black"].get("user", {}).get("name") or ev["black"].get("id")
                        our_color = None
                        if our_id and white_id and our_id.lower() == white_id.lower():
                            our_color = chess.WHITE
                        elif our_id and black_id and our_id.lower() == black_id.lower():
                            our_color = chess.BLACK

                        # If it's our turn, ask agent for move
                        try:
                            if board.turn == our_color:
                                move = self.agent.select_move(board)
                                if move is not None:
                                    ok = self.send_move(game_id, move.uci())
                                    if not ok:
                                        print(f"Failed to send move {move} for game {game_id}")
                        except Exception as e:
                            print("Error selecting/sending move:", e)

    def run(self):
        # Start listening to Lichess events and spawn handlers for games we are in
        print("Starting Lichess adapter event stream...")
        for ev in self.stream_events():
            if self._stop:
                break
            t = ev.get("type")
            if t == "challenge":
                # auto-decline challenges (or implement acceptance logic)
                chal_id = ev.get("challenge", {}).get("id")
                if chal_id:
                    self.session.post(f"{self.base}/api/challenge/{chal_id}/decline")
            elif t == "gameStart":
                game_id = ev.get("game", {}).get("id")
                if game_id:
                    print(f"Starting handler for game {game_id}")
                    th = threading.Thread(target=self.handle_game, args=(game_id,), daemon=True)
                    th.start()

    def stop(self):
        self._stop = True
