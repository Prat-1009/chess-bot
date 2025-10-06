import pytest
from chess_bot.adapter_chesscom import parse_moves_text


def test_parse_simple_moves():
    txt = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6"
    tokens = parse_moves_text(txt)
    assert tokens == ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6']


def test_parse_with_clocks_and_annotations():
    txt = "1. e4 0.9s e5 14.2s 2. f4 2.5s"
    tokens = parse_moves_text(txt)
    assert tokens == ['e4', 'e5', 'f4']


def test_parse_with_uci_and_punctuation():
    txt = "e2e4, e7e5; g1f3"
    tokens = parse_moves_text(txt)
    assert tokens == ['e2e4', 'e7e5', 'g1f3']
