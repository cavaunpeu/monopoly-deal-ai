#!/usr/bin/env python3
"""
CLI tool for understanding action mappings in CFR checkpoints.
"""

import sys

from game.action import IDX_TO_ABSTRACT_ACTION


def print_action_mapping() -> None:
    """Print the mapping from action indices to AbstractAction enum values."""
    print("Action Index to AbstractAction Mapping:")
    print("=" * 40)

    for i, action in IDX_TO_ABSTRACT_ACTION.items():
        print(f"{i:2d} -> {action.name}")

    print(f"\nTotal actions: {len(IDX_TO_ABSTRACT_ACTION)}")


def main() -> None:
    """Main entry point for the inspection CLI tool."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m models.cfr.inspect actions                    # Show action mapping")
        sys.exit(1)

    command = sys.argv[1]

    if command == "actions":
        print_action_mapping()
    else:
        print(f"Error: Unknown command '{command}'")
        print("Available commands: actions")
        sys.exit(1)


if __name__ == "__main__":
    main()
