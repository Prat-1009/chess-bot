import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import numpy as np
import random
from typing import List, Tuple


def board_to_tensor(board: chess.Board) -> torch.FloatTensor:
    """Convert board to a simple tensor of shape (12, 8, 8) representing piece planes.
    Order: P,N,B,R,Q,K for white then same for black.
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        plane = None
        if piece.piece_type == chess.PAWN:
            base = 0
        elif piece.piece_type == chess.KNIGHT:
            base = 1
        elif piece.piece_type == chess.BISHOP:
            base = 2
        elif piece.piece_type == chess.ROOK:
            base = 3
        elif piece.piece_type == chess.QUEEN:
            base = 4
        elif piece.piece_type == chess.KING:
            base = 5
        plane = base + (0 if piece.color == chess.WHITE else 6)
        row = 7 - (square // 8)
        col = square % 8
        planes[plane, row, col] = 1.0
    return torch.from_numpy(planes)


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        # Output: logits over all possible moves encoded as uci string index
        # We'll map moves to indices dynamically per position
        self.fc2 = nn.Linear(512, 4096)  # 4096 > typical legal moves upper bound

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class RLAgent:
    def __init__(self, lr: float = 1e-3, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PolicyNet().to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def select_move(self, board: chess.Board) -> chess.Move:
        self.net.eval()
        with torch.no_grad():
            state = board_to_tensor(board).unsqueeze(0).to(self.device)
            logits = self.net(state).squeeze(0).cpu().numpy()
            # Map legal moves to logits
            legal = list(board.legal_moves)
            if len(legal) == 0:
                return None
            move_indices = []
            move_scores = []
            for move in legal:
                # simple hash: python-chess move.uci() -> map to index via hashing
                idx = (hash(move.uci()) % logits.shape[0])
                move_indices.append((move, idx))
                move_scores.append(logits[idx])
            probs = np.exp(np.array(move_scores) - np.max(move_scores))
            probs = probs / probs.sum()
            choice = np.random.choice(len(legal), p=probs)
            return legal[choice]

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))


class Trainer:
    def __init__(self, agent: RLAgent, gamma: float = 0.99):
        self.agent = agent
        self.gamma = gamma

    def play_episode_selfplay(self, max_moves: int = 200) -> Tuple[List[Tuple[torch.Tensor, int]], float]:
        """
        Play self-play: the agent plays both sides. Record log_probs together with which color made the move
        (1 for white, -1 for black). Return list of (log_prob, color) and final reward for white (1.0 win, 0 else).
        """
        board = chess.Board()
        records: List[Tuple[torch.Tensor, int]] = []
        while not board.is_game_over() and board.fullmove_number < max_moves:
            # Agent plays for current side
            log_prob = self.agent_select_and_record(board)
            if log_prob is None:
                break
            color = 1 if board.turn == chess.WHITE else -1
            records.append((log_prob, color))
            # push already done inside agent_select_and_record
        # Outcome from perspective of white
        result = board.result()
        reward_white = 1.0 if result == '1-0' else 0.0
        return records, reward_white

    def agent_select_and_record(self, board: chess.Board):
        """Select a move for the agent and record its log_prob, then push the move on the board."""
        self.agent.net.train()
        state = board_to_tensor(board).unsqueeze(0).to(self.agent.device)
        logits = self.agent.net(state).squeeze(0)
        legal = list(board.legal_moves)
        if len(legal) == 0:
            return None
        move_scores = []
        for move in legal:
            idx = (hash(move.uci()) % logits.shape[0])
            move_scores.append(logits[idx])
        scores = torch.stack(move_scores).to(self.agent.device)
        probs = torch.softmax(scores, dim=0)
        m = torch.distributions.Categorical(probs)
        choice = m.sample()
        chosen_move = legal[choice.item()]
        log_prob = m.log_prob(choice)
        board.push(chosen_move)
        return log_prob

    def update_policy_selfplay(self, records: List[Tuple[torch.Tensor, int]], reward_white: float):
        """
        Update policy using REINFORCE where each log_prob receives reward depending on which color played it.
        If white wins: white moves get +1, black moves get -1 (so agent favors white); if black wins or draw: reward 0.
        """
        if len(records) == 0:
            return
        losses = []
        for log_prob, color in records:
            # assign reward for this move: white_reward * color
            r = reward_white * color
            losses.append(-log_prob * r)
        loss = torch.stack(losses).sum()
        self.agent.opt.zero_grad()
        loss.backward()
        self.agent.opt.step()

    def train(self, episodes: int = 10, max_moves: int = 200, verbose: bool = True):
        for ep in range(episodes):
            records, reward_white = self.play_episode_selfplay(max_moves=max_moves)
            self.update_policy_selfplay(records, reward_white)
            if verbose:
                print(f"Episode {ep+1}/{episodes} reward_white={reward_white} moves={len(records)}")
