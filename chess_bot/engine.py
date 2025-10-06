import chess
import math

class SimpleEngine:
    """A simple minimax engine with alpha-beta pruning.

    - evaluate: material count
    - search: minimax with depth
    """

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }

    def __init__(self, depth=2):
        self.depth = depth

    def evaluate(self, board: chess.Board) -> int:
        """Simple material evaluation from side-to-move perspective.
        Positive means advantage for white.
        """
        score = 0
        for piece_type in self.PIECE_VALUES:
            value = self.PIECE_VALUES[piece_type]
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        # Perspective: if black to move, invert
        return score if board.turn == chess.WHITE else -score

    def search(self, board: chess.Board) -> chess.Move:
        """Return best move found from root using alpha-beta."""
        best_move = None
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf
        for move in board.legal_moves:
            board.push(move)
            score = -self._alphabeta(board, self.depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
        return best_move

    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)
        value = -math.inf
        for move in board.legal_moves:
            board.push(move)
            score = -self._alphabeta(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value
