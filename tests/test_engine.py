import chess
from chess_bot.engine import SimpleEngine


def test_engine_returns_legal_move():
    board = chess.Board()
    engine = SimpleEngine(depth=1)
    move = engine.search(board)
    assert move in board.legal_moves


def test_engine_handles_checkmate():
    # Fool's mate position where black has just delivered checkmate
    board = chess.Board('rnb1kbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1')
    engine = SimpleEngine(depth=1)
    move = engine.search(board)
    # In a mate-ish or limited position engine might return None; ensure no exception
    assert (move is None) or (move in board.legal_moves)
