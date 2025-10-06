"""
Playwright-based adapter for chess.com (experimental).

WARNING: Automating chess.com may violate their Terms of Service. Use only on accounts you own and preferably test accounts.

This adapter is best-effort and fragile: page structure can change. It provides a dry-run parsing mode (recommended) and a best-effort auto-play mode that simulates drag-and-drop.

Usage pattern:
  from chess_bot.adapter_chesscom import ChessComAdapter
  adapter = ChessComAdapter(agent)
  adapter.run(username, password, game_url, dry_run=True)

"""

from playwright.sync_api import sync_playwright, Page
import chess
import time
import math
import logging
import io
import random
import re
import os
from datetime import datetime
from typing import Optional, Tuple, List, Callable

logger = logging.getLogger(__name__)


def retry_backoff(max_attempts: int = 4, base_delay: float = 0.5, jitter: float = 0.1):
    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.exception("Operation failed after %d attempts", attempt)
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    delay = delay + random.uniform(-jitter, jitter)
                    logger.warning("Operation failed (attempt %d/%d): %s; retrying in %.2fs",
                                   attempt, max_attempts, e, delay)
                    time.sleep(max(0.0, delay))
        return wrapper
    return decorator


def san_list_from_page(page: Page) -> List[str]:
    """Extract SAN/Move tokens from the chess.com game page.

    This is a best-effort parser that tries several selectors and falls back to PGN.
    Returns a list of SAN or UCI tokens (may be empty).
    """
    selectors = [
        '.vertical-move-list', '.move-list', '.game-moves', '.moves', '.move-text',
        '.moves-list', '.move-history', '.move-history-list', '.live-game-move-list'
    ]

    # Try to parse elements with individual move nodes
    for sel in selectors:
        try:
            node = page.query_selector(sel)
            if not node:
                continue
            text = (node.inner_text() or '').strip()
            if not text:
                continue
            moves = parse_moves_text(text)
            if moves:
                logger.debug('san_list_from_page: parsed %d moves from %s', len(moves), sel)
                return moves
        except Exception:
            continue

    # Fallback: try to read PGN
    try:
        pgn_el = page.query_selector('textarea.pgn') or page.query_selector('.pgn')
        if pgn_el:
            try:
                pgn = pgn_el.inner_text() or pgn_el.input_value()
                if pgn:
                    import chess.pgn
                    g = chess.pgn.read_game(io.StringIO(pgn))
                    if g is not None:
                        moves = [node.san() for node in g.mainline() if node.san()]
                        logger.debug('san_list_from_page: parsed %d moves from PGN', len(moves))
                        return moves
            except Exception:
                logger.exception('Failed to parse PGN from page')
    except Exception:
        pass

    logger.debug('san_list_from_page: no moves found')
    return []


def parse_moves_text(text: str) -> List[str]:
    """Parse a block of move-list text and return only valid SAN/UCI move tokens.

    This filters out move numbers, game results, clocks (e.g. '0.9s'), and stray annotations.
    """
    if not text:
        return []
    # split on whitespace, but many pages include times next to moves
    toks = [t.strip() for t in text.split() if t.strip()]
    # regexes
    move_num_re = re.compile(r'^\d+\.$')
    clock_re = re.compile(r'^\d+(?:\.\d+)?s$')
    result_set = {'1-0', '0-1', '1/2-1/2', '*'}
    # SAN and simple UCI acceptance regex
    san_re = re.compile(r'^(O-O(?:-O)?|[PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?|[a-h][1-8])$')
    uci_re = re.compile(r'^[a-h][1-8][a-h][1-8][nbrqNBRQ]?$')

    moves: List[str] = []
    for t in toks:
        # strip common annotations
        tt = t.replace('++', '').replace('+', '').replace('!', '').replace('?', '').strip()
        if not tt:
            continue
        if move_num_re.match(tt):
            continue
        if tt in result_set:
            continue
        if clock_re.match(tt):
            continue
        # accept SAN or UCI
        if san_re.match(tt) or uci_re.match(tt):
            moves.append(tt)
            continue
        # sometimes tokens are like 'e4,' or 'e4;' -- strip punctuation
        clean = tt.strip('.,;:')
        if san_re.match(clean) or uci_re.match(clean):
            moves.append(clean)
            continue
        # ignore anything else
    return moves


def build_board_from_san_list(sans: List[str]) -> chess.Board:
    board = chess.Board()
    for san in sans:
        try:
            board.push_san(san)
        except Exception:
            # ignore unparsable SAN tokens
            try:
                # attempt to interpret as UCI
                board.push_uci(san)
            except Exception:
                pass
    return board


class ChessComAdapter:
    def __init__(self, agent, headless: bool = True):
        self.agent = agent
        self.headless = headless

    def _find_board_element(self, page: Page):
        # Common selectors for chess.com board container; check several common names
        candidates = [
            'div.board', 'chess-board', 'div.board-component', '.board', '.board-wrap',
            '.board-layout', '#board', '.board__board', 'div[data-role="board"]', '.board-area'
        ]
        for sel in candidates:
            try:
                el = page.query_selector(sel)
                if el:
                    return el
            except Exception:
                continue
        # fallback: look for an element that contains squares or data-square attributes
        try:
            sq = page.query_selector('[data-square]') or page.query_selector('.square')
            if sq:
                # return parent element as a heuristic
                parent = sq.evaluate_handle('el => el.closest("[data-role=board], .board, .board-area, .board__board")')
                try:
                    if parent:
                        return parent
                except Exception:
                    pass
        except Exception:
            pass
        # nothing found: write debug screenshot and page HTML to help diagnosis
        try:
            dump_dir = os.path.abspath('debug_screenshots')
            os.makedirs(dump_dir, exist_ok=True)
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
            png = os.path.join(dump_dir, f'no_board_{ts}.png')
            htmlf = os.path.join(dump_dir, f'no_board_{ts}.html')
            try:
                page.screenshot(path=png, full_page=True)
            except Exception:
                pass
            try:
                with open(htmlf, 'w', encoding='utf-8') as fh:
                    fh.write(page.content())
            except Exception:
                pass
            logger.warning('Board not found; saved debug screenshot and HTML to %s', dump_dir)
        except Exception:
            logger.exception('Failed to write debug dump')
        return None

    def _square_center(self, board_box: dict, square: str, flipped: bool) -> Tuple[float, float]:
        # board_box: dict from bounding_box() with x,y,width,height
        file = ord(square[0]) - ord('a')  # 0..7
        rank = int(square[1])  # 1..8
        square_w = board_box['width'] / 8.0
        square_h = board_box['height'] / 8.0
        if not flipped:
            # a1 is bottom-left -> pixel coords
            x = board_box['x'] + (file + 0.5) * square_w
            # y coordinate: top is a8 -> y = y + (8 - rank + 0.5) * square_h
            y = board_box['y'] + (8 - rank + 0.5) * square_h
        else:
            # flipped: a1 is top-right
            x = board_box['x'] + ((7 - file) + 0.5) * square_w
            y = board_box['y'] + (rank - 1 + 0.5) * square_h
        return x, y

    def _ensure_overlay(self, page: Page):
        # Inject a simple overlay div and marker containers if not present
        js = r"""
        if (!document.getElementById('cp_overlay')) {
            const d = document.createElement('div');
            d.id = 'cp_overlay';
            d.style.position = 'fixed';
            d.style.right = '12px';
            d.style.top = '12px';
            d.style.zIndex = 9999999;
            d.style.background = 'rgba(0,0,0,0.7)';
            d.style.color = 'white';
            d.style.padding = '8px 12px';
            d.style.borderRadius = '6px';
            d.style.fontFamily = 'Arial, sans-serif';
            d.style.fontSize = '14px';
            d.innerText = 'Suggestions: initializing...';
            document.body.appendChild(d);
        }
        if (!document.getElementById('cp_from_marker')) {
            const m1 = document.createElement('div');
            m1.id = 'cp_from_marker';
            m1.style.position = 'absolute';
            m1.style.width = '18px';
            m1.style.height = '18px';
            m1.style.borderRadius = '50%';
            m1.style.background = 'rgba(0,200,0,0.9)';
            m1.style.zIndex = 9999998;
            m1.style.pointerEvents = 'none';
            document.body.appendChild(m1);
        }
        if (!document.getElementById('cp_to_marker')) {
            const m2 = document.createElement('div');
            m2.id = 'cp_to_marker';
            m2.style.position = 'absolute';
            m2.style.width = '18px';
            m2.style.height = '18px';
            m2.style.borderRadius = '50%';
            m2.style.background = 'rgba(200,0,0,0.9)';
            m2.style.zIndex = 9999998;
            m2.style.pointerEvents = 'none';
            document.body.appendChild(m2);
        }
        // create arrow svg container
        if (!document.getElementById('cp_arrow_svg')) {
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('id', 'cp_arrow_svg');
            svg.style.position = 'absolute';
            svg.style.left = '0px';
            svg.style.top = '0px';
            svg.style.width = '100%';
            svg.style.height = '100%';
            svg.style.zIndex = 9999997;
            svg.style.pointerEvents = 'none';
            document.body.appendChild(svg);
        }
        // create confirm modal
        if (!document.getElementById('cp_confirm')) {
            const modal = document.createElement('div');
            modal.id = 'cp_confirm';
            modal.style.position = 'fixed';
            modal.style.left = '50%';
            modal.style.top = '50%';
            modal.style.transform = 'translate(-50%,-50%)';
            modal.style.zIndex = 10000000;
            modal.style.background = 'rgba(0,0,0,0.8)';
            modal.style.color = 'white';
            modal.style.padding = '12px';
            modal.style.borderRadius = '8px';
            modal.style.display = 'none';
            const txt = document.createElement('div');
            txt.id = 'cp_confirm_text';
            modal.appendChild(txt);
            const btnY = document.createElement('button');
            btnY.innerText = 'Confirm';
            btnY.style.margin = '6px';
            btnY.onclick = function(){ window.__cp_confirm_result = true; modal.style.display='none'; };
            const btnN = document.createElement('button');
            btnN.innerText = 'Cancel';
            btnN.style.margin = '6px';
            btnN.onclick = function(){ window.__cp_confirm_result = false; modal.style.display='none'; };
            modal.appendChild(btnY);
            modal.appendChild(btnN);
            document.body.appendChild(modal);
            window.__cp_confirm_result = null;
        }
        """
        try:
            page.evaluate(js)
        except Exception:
            logger.exception('Failed to inject overlay')

    def _update_overlay(self, page: Page, from_pos: Tuple[float, float], to_pos: Tuple[float, float], text: str):
        # Place the overlay text and move markers at given pixel positions
        fx, fy = from_pos
        tx, ty = to_pos
        js = f"""
        (function(){{
            const d = document.getElementById('cp_overlay');
            if (d) d.innerText = `{text}`;
            const f = document.getElementById('cp_from_marker');
            const t = document.getElementById('cp_to_marker');
            if (f) {{ f.style.left = '{fx}px'; f.style.top = '{fy}px'; }}
            if (t) {{ t.style.left = '{tx}px'; t.style.top = '{ty}px'; }}
        }})();
        """
        try:
            page.evaluate(js)
        except Exception:
            logger.exception('Failed to update overlay')

    def _draw_arrow(self, page: Page, from_pos: Tuple[float, float], to_pos: Tuple[float, float], color: str = 'yellow'):
        fx, fy = from_pos
        tx, ty = to_pos
        js = f"""
        (function(){{
            const svg = document.getElementById('cp_arrow_svg');
            if (!svg) return;
            // clear existing
            while (svg.firstChild) svg.removeChild(svg.firstChild);
            const ns = 'http://www.w3.org/2000/svg';
            const line = document.createElementNS(ns, 'line');
            line.setAttribute('x1', '{fx}');
            line.setAttribute('y1', '{fy}');
            line.setAttribute('x2', '{tx}');
            line.setAttribute('y2', '{ty}');
            line.setAttribute('stroke', '{color}');
            line.setAttribute('stroke-width', '4');
            line.setAttribute('stroke-linecap', 'round');
            svg.appendChild(line);
            // arrow head
            const head = document.createElementNS(ns, 'polygon');
            const dx = ({tx} - {fx});
            const dy = ({ty} - {fy});
            const len = Math.sqrt(dx*dx + dy*dy) || 1;
            const ux = dx/len; const uy = dy/len;
            const px = {tx} - ux*12; const py = {ty} - uy*12;
            const leftx = px - uy*6; const lefty = py + ux*6;
            const rightx = px + uy*6; const righty = py - ux*6;
            head.setAttribute('points', `${{ {tx},{ty} }} ${{ leftx }},${{ lefty }} ${{ rightx }},${{ righty }}`);
            head.setAttribute('fill', '{color}');
            svg.appendChild(head);
        }})();
        """
        try:
            page.evaluate(js)
        except Exception:
            logger.exception('Failed to draw arrow')

    def _confirm_in_page(self, page: Page, text: str, timeout: float = 10.0) -> bool:
        # show modal and wait for window.__cp_confirm_result to be set
        js_show = f"document.getElementById('cp_confirm_text').innerText = `{text}`; document.getElementById('cp_confirm').style.display = 'block'; window.__cp_confirm_result = null;"
        try:
            page.evaluate(js_show)
        except Exception:
            logger.exception('Failed to show confirm modal')
            return False
        # poll for result
        start = time.time()
        while time.time() - start < timeout:
            try:
                res = page.evaluate('window.__cp_confirm_result')
                if res is True:
                    return True
                if res is False:
                    return False
            except Exception:
                pass
            time.sleep(0.2)
        # timeout
        try:
            page.evaluate('document.getElementById(\'cp_confirm\').style.display = \"none\"; window.__cp_confirm_result = null;')
        except Exception:
            pass
        return False

    def _detect_flipped(self, board_el) -> bool:
        try:
            cl = board_el.get_attribute('class') or ''
            if 'flipped' in cl:
                return True
            # data-orientation attribute
            orient = board_el.get_attribute('data-orientation')
            if orient and orient.lower() == 'black':
                return True
        except Exception:
            pass
        return False

    def apply_move_via_drag(self, page: Page, board_el, from_sq: str, to_sq: str) -> bool:
        @retry_backoff(max_attempts=3, base_delay=0.3)
        def _do_drag():
            box = board_el.bounding_box()
            if not box:
                raise RuntimeError('no bounding box')
            flipped = self._detect_flipped(board_el)
            fx, fy = self._square_center(box, from_sq, flipped)
            tx, ty = self._square_center(box, to_sq, flipped)
            page.mouse.move(fx, fy)
            page.mouse.down()
            page.mouse.move(tx, ty, steps=10)
            page.mouse.up()
            return True

        try:
            return _do_drag()
        except Exception as e:
            logger.exception('apply_move_via_drag failed: %s', e)
            return False

    def run(self, username: str, password: str, game_url: str, dry_run: bool = True, headful: bool = False, browser_channel: str = None, show_suggestions: bool = False, auto_confirm: bool = False, storage_state: str = None):
        with sync_playwright() as p:
            launch_args = {}
            if browser_channel:
                # Use Playwright 'channel' to launch installed Chrome/Edge etc.
                launch_args['channel'] = browser_channel
                logger.info('Launching browser with channel %s', browser_channel)
            browser = p.chromium.launch(headless=(not headful), **launch_args)
            # Load storage_state if provided and exists
            context_args = {}
            if storage_state and os.path.exists(storage_state):
                try:
                    context_args['storage_state'] = storage_state
                    logger.info('Loading Playwright storage_state from %s', storage_state)
                except Exception:
                    logger.exception('Failed to load storage_state file')
            context = browser.new_context(**context_args)
            page = context.new_page()

            @retry_backoff(max_attempts=3, base_delay=0.5)
            def goto_url(u: str):
                page.goto(u)
                page.wait_for_load_state('networkidle')

            # login (best-effort) - do not fail hard if login elements differ
            try:
                goto_url('https://www.chess.com/login')
                # try common input names
                possible_username = ['input#username', 'input[name=username]', 'input[name=user]']
                possible_password = ['input#password', 'input[name=password]']
                u_sel = next((s for s in possible_username if page.query_selector(s)), None)
                p_sel = next((s for s in possible_password if page.query_selector(s)), None)
                if u_sel and p_sel and username and password:
                    logger.info('Attempting login using selectors %s / %s', u_sel, p_sel)
                    page.fill(u_sel, username, timeout=5000)
                    page.fill(p_sel, password, timeout=5000)
                    # click the first submit button we find
                    btn = page.query_selector('button[type=submit]')
                    if btn:
                        btn.click()
                    page.wait_for_load_state('networkidle', timeout=10000)
                else:
                    logger.info('Login selectors not found or credentials not provided; continuing (maybe session exists)')
                # After attempting login (or skipping), if storage_state path provided, save current storage state
                try:
                    if storage_state:
                        # attempt to persist context state so subsequent runs avoid re-login
                        logger.info('Saving Playwright storage_state to %s', storage_state)
                        context.storage_state(path=storage_state)
                except Exception:
                    logger.exception('Failed to save storage_state')
            except Exception as e:
                logger.warning('Login attempt failed (continuing): %s', e)

            # Go to game URL
            try:
                goto_url(game_url)
            except Exception as e:
                logger.exception('Failed to open game URL: %s', e)
                return

            board_el = self._find_board_element(page)
            if not board_el:
                logger.warning('Could not find board element on page; adapter may be incompatible with this layout')

            # main loop: poll moves every few seconds
            poll_interval = 2.0
            while True:
                try:
                    sans = san_list_from_page(page)
                    logger.debug('Parsed SAN tokens: %s', sans)
                    board = build_board_from_san_list(sans)
                    logger.debug('Reconstructed board FEN: %s', board.fen())
                    # Detect our color robustly: look for .player-username or similar elements
                    our_color = None
                    try:
                        # look for username labels in different parts of the page
                        w_el = page.query_selector('.player-username.white, .username.white, .player-name.white')
                        b_el = page.query_selector('.player-username.black, .username.black, .player-name.black')
                        white_name = (w_el.inner_text() or '').strip().lower() if w_el else None
                        black_name = (b_el.inner_text() or '').strip().lower() if b_el else None
                        uname = (username or '').strip().lower()
                        if uname and white_name and uname in white_name:
                            our_color = chess.WHITE
                        elif uname and black_name and uname in black_name:
                            our_color = chess.BLACK
                    except Exception:
                        pass
                    if our_color is None:
                        # fallback heuristics: if moves length is even assume white to move else black
                        our_color = chess.WHITE
                    logger.debug('Detected our_color=%s (WHITE=1, BLACK=0)', 'WHITE' if our_color==chess.WHITE else 'BLACK')

                    # If it's our turn
                    if board.turn == our_color:
                        logger.info('It is our turn (board.turn=%s)', 'WHITE' if board.turn==chess.WHITE else 'BLACK')
                        try:
                            move = self.agent.select_move(board)
                        except Exception as e:
                            logger.exception('Agent.select_move raised exception: %s', e)
                            move = None
                        if move is None:
                            logger.info('Agent returned no move')
                        else:
                            logger.info('Agent suggests %s', move.uci())
                            # show overlay if requested
                            if show_suggestions and board_el:
                                try:
                                    # ensure overlay exists
                                    self._ensure_overlay(page)
                                    box = board_el.bounding_box()
                                    flipped = self._detect_flipped(board_el)
                                    fx, fy = self._square_center(box, move.uci()[:2], flipped)
                                    tx, ty = self._square_center(box, move.uci()[2:], flipped)
                                    # convert to ints and offset a bit to center markers
                                    self._update_overlay(page, (int(fx)-9, int(fy)-9), (int(tx)-9, int(ty)-9), f'Suggested: {move.uci()}')
                                except Exception:
                                    logger.exception('Failed to update suggestion overlay')
                            if not dry_run and board_el:
                                do_apply = True
                                if show_suggestions and not auto_confirm:
                                    # ask for confirmation in-page
                                    try:
                                        okc = self._confirm_in_page(page, f'Apply move {move.uci()}?', timeout=12.0)
                                        logger.debug('Confirm modal returned: %s', okc)
                                        if not okc:
                                            logger.info('User declined move %s', move.uci())
                                            do_apply = False
                                    except Exception:
                                        logger.exception('Confirm modal failed; skipping confirmation')
                                if do_apply:
                                    try:
                                        ok = self.apply_move_via_drag(page, board_el, move.uci()[:2], move.uci()[2:])
                                        logger.info('Applied via drag: %s', ok)
                                        if not ok:
                                            logger.warning('apply_move_via_drag returned False for move %s', move.uci())
                                    except Exception as e:
                                        logger.exception('Error during apply_move_via_drag: %s', e)
                    time.sleep(poll_interval)
                except KeyboardInterrupt:
                    logger.info('Adapter interrupted by user')
                    break
                except Exception as e:
                    logger.exception('Adapter loop error: %s', e)
                    time.sleep(3.0)
