import logging
import os
import re
import sys
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import chess
import chess.engine
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import datasets
from datasets import Dataset
import transformers
from transformers import set_seed, default_data_collator
from transformers.trainer_utils import get_last_checkpoint

from trl import GRPOTrainer, GRPOConfig, ScriptArguments, TrlParser, get_peft_config

# ------------------------------
# Logging Setup
# ------------------------------
logger = logging.getLogger(__name__)

# ------------------------------
# Dataset Formatting and Collators
# ------------------------------
def format_dataset(dataset):
    """Ensures dataset samples are dictionaries instead of lists."""
    return dataset.map(lambda example: {k: v for k, v in example.items()})

def identity_collate_fn(examples):
    """
    Identity collator: returns the list of examples as is.
    This ensures that during training GRPOTrainer receives a list of dictionaries.
    """
    return examples

# ------------------------------
# Chess Engine and Reward Functions
# ------------------------------
ENGINE_PATH = "/usr/games/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
ANALYSIS_TIME = 0.1  # seconds
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User provides a chess position, and the Assistant determines the best move. "
    "The Assistant first thinks about the reasoning process and then provides the User with the move. The reasoning process "
    "is enclosed within <think> </think> tags, and the move is enclosed within <move> </move> tags, i.e., "
    "<think> reasoning process here </think><move> best move in UCI format here </move>."
)

# For tracking loss values
loss_values = []

def log_loss_to_file(loss, step):
    """Logs loss to a CSV file."""
    with open("training_loss.csv", "a") as f:
        f.write(f"{step},{loss}\n")

# ------------------------------
# Model and Data Arguments
# ------------------------------
@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_peft: bool = field(default=True, metadata={"help": "Whether to use PEFT or not"})
    lora_r: int = field(default=16, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=32, metadata={"help": "Alpha parameter for LoRA scaling"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Dropout probability for LoRA layers"})

@dataclass
class DataArguments:
    """Arguments for data generation and processing."""
    num_positions: int = field(
        default=1000,
        metadata={"help": "Number of chess positions to generate for training"}
    )
    data_output_dir: str = field(
        default="./chess_model",
        metadata={"help": "Output directory for dataset and generated files"}
    )

# ------------------------------
# Chess Functions
# ------------------------------
def get_top_moves(board: chess.Board, num_moves: int = 20) -> List[Tuple[chess.Move, float]]:
    """Get the top N moves for a position according to Stockfish."""
    moves = []
    try:
        # Use multipv to get multiple lines of analysis
        info = engine.analyse(
            board,
            chess.engine.Limit(time=ANALYSIS_TIME),
            multipv=num_moves
        )
        # Extract moves and scores
        for pv_info in info:
            if 'pv' in pv_info and len(pv_info['pv']) > 0:
                move = pv_info['pv'][0]
                # Using mate_score=10000 as a fallback for mate values
                score = pv_info['score'].relative.score(mate_score=10000)
                moves.append((move, score))
    except Exception as e:
        logger.warning(f"Error analyzing position: {e}")
        return []
    # Sort moves by score descending and return the top moves
    moves.sort(key=lambda x: x[1], reverse=True)
    return moves[:num_moves]

def create_chess_dataset(num_positions: int) -> Dataset:
    """
    Create a dataset of chess positions for training.
    For each position, we get the top moves (using Stockfish) and generate an example with the system prompt, FEN, best move, and its score.
    """
    data = []
    board = chess.Board()
    positions_seen = set()  # Track unique positions

    while len(data) < num_positions:
        print(f"Generated examples: {len(data)}", end="\r")
        fen = board.fen()

        # Skip if this position has already been seen or if the game is over
        if fen in positions_seen or board.is_game_over():
            board = chess.Board()  # Reset board
            continue

        positions_seen.add(fen)

        # Get top moves for this position
        top_moves = get_top_moves(board)
        # Create examples for each top move
        for move, score in top_moves:
            example = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Playing as {'white' if board.turn else 'black'}, make a move in this position: {fen}"}
                ],
                "fen": fen,
                "best_move": move.uci(),
                "move_score": score
            }
            data.append(example)
            if len(data) >= num_positions:
                break

        # Continue game by making one random move from the top moves (if available)
        if top_moves:
            selected_move = random.choice(top_moves)[0]
            board.push(selected_move)
    print()  # Newline after progress printing
    return Dataset.from_list(data[:num_positions])

def evaluate_move(board: chess.Board, move_uci: str) -> float:
    """Evaluate the quality of a given move using the chess engine."""
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return -1.0
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
        board.pop()
        score = info.get("score")
        if score is None:
            return 0.0
        # Convert mate scores to centipawn scores; note that score.white() converts for white's perspective
        cp_score = score.white().score(mate_score=10000)
        # Normalize score to [0, 1] using a sigmoid-like function
        normalized_score = 1 / (1 + pow(10, -cp_score/400))
        return normalized_score
    except (ValueError, chess.IllegalMoveError):
        return -1.0

def structure_reward(completions, **kwargs):
    """
    Reward function that checks if the completion follows the required format:
    <think> ... </think><move> ... </move>
    """
    pattern = r"^<think>.*?</think><move>[a-h][1-8][a-h][1-8]</move>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [bool(re.match(pattern, content)) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def move_quality_reward(completions, fen, best_move, **kwargs):
    """
    Reward function that evaluates the quality of the chosen move.
    It gives full reward if the move exactly matches the best move;
    otherwise, it evaluates the move using the chess engine.
    """
    rewards = []
    # Ensure `fen` is a string
    if isinstance(fen, list):
        fen = fen[0]
    for completion in completions:
        content = completion[0]["content"]
        board = chess.Board(fen)
        move_match = re.search(r"<move>([a-h][1-8][a-h][1-8])</move>", content)
        if not move_match:
            rewards.append(0.0)
            continue
        move_uci = move_match.group(1)
        if move_uci == best_move:
            rewards.append(1.0)
        else:
            reward = evaluate_move(board, move_uci)
            rewards.append(reward)
    return rewards

reward_funcs_registry = {
    "structure": structure_reward,
    "move_quality": move_quality_reward,
}

# ------------------------------
# Custom Trainer Subclass
# ------------------------------
from torch.utils.data import DataLoader

class CustomGRPOTrainer(GRPOTrainer):
    """
    A custom subclass of GRPOTrainer that overrides the dataloader methods
    to use an identity collator for training (returning a list of dictionaries)
    and the default collator for evaluation (returning a dictionary).
    """
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=identity_collate_fn,
        )
    
    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=default_data_collator,  # Returns a dictionary
        )

# ------------------------------
# Main Function and Trainer Setup
# ------------------------------
def main():
    # Parse arguments using Hugging Face's HfArgumentParser
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GRPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Create chess dataset
    logger.info("Creating chess dataset...")
    dataset = create_chess_dataset(data_args.num_positions)
    train_size = int(len(dataset) * 0.8)
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    train_dataset = format_dataset(train_dataset)
    eval_dataset = format_dataset(eval_dataset)

    # Optional: Print the first sample for debugging
    logger.info(f"First training example: {train_dataset[0]}")

    # Configure PEFT (if applicable)
    peft_config = None  # Adjust as needed if using PEFT

    # Get reward functions to pass to the trainer
    reward_funcs = [
        reward_funcs_registry["structure"],
        reward_funcs_registry["move_quality"]
    ]

    # Initialize our custom trainer (which subclasses GRPOTrainer)
    trainer = CustomGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    logger.info("Starting training...")

    # Training loop (assuming trainer.train() yields (step, loss))
    for step, loss in enumerate(trainer.train()):
        print(f"Step {step}: Loss = {loss:.4f}")
        loss_values.append(loss)
        log_loss_to_file(loss, step)
        if step % 10 == 0:
            logger.info(f"Step {step}: Loss = {loss:.4f}")

    # Save loss values to a CSV file
    loss_df = pd.DataFrame({"step": list(range(len(loss_values))), "loss": loss_values})
    loss_df.to_csv("training_loss.csv", index=False)

    # (Optional) Plot the loss curve after training
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")

    # Save the final model
    trainer.save_model(training_args.data_output_dir)

    # Clean up the chess engine
    engine.quit()

if __name__ == "__main__":
    main()
