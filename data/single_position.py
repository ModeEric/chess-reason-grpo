import chess
import chess.engine
import json

# Adjust the path to point to your UCI engine executable
ENGINE_PATH = "/usr/local/bin/stockfish"  # e.g., "/usr/local/bin/stockfish"

# Initialize engine (set a short analysis time per move)
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
analysis_time = 0.1  # seconds

def evaluate_position(board):
    """
    Evaluate all legal moves for a given board position.
    Returns a dictionary mapping moves (in UCI format) to their evaluation scores.
    """
    move_evaluations = {}
    
    for move in board.legal_moves:
        board.push(move)  # Make the move temporarily
        
        # Analyze the position after the move.
        # The score returned is from the perspective of the side to move in that position.
        info = engine.analyse(board, chess.engine.Limit(time=analysis_time))
        score = info.get("score")
        
        # Convert the score to a centipawn value (handling mate scores).
        # Here, mate_score is set arbitrarily high (e.g., 10000) to differentiate mate in n moves.
        if score is not None:
            cp_score = score.white().score(mate_score=10000)
        else:
            cp_score = None
        
        move_evaluations[move.uci()] = cp_score
        board.pop()  # Undo the move
        
    return move_evaluations

# Example usage:
if __name__ == "__main__":
    # Start from the standard initial position
    board = chess.Board()
    
    # Optionally, you could load a position via FEN:
    # board.set_fen("r1bqkbnr/pppppppp/n7/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Evaluate the current position
    move_eval_map = evaluate_position(board)
    
    # Get the board position in FEN format
    position_fen = board.fen()
    
    # Print the result (or store it in your dataset)
    print("Board Position (FEN):", position_fen)
    print("Move Evaluations:", json.dumps(move_eval_map, indent=2))
    
    # Don't forget to close the engine when done
    engine.quit()
