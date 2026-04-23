#!/usr/bin/env bash
# Thin wrapper around the CLI for common ingest workflows.
#
# Usage:
#   ./ingest.sh health
#   ./ingest.sh create  books-fresh
#   ./ingest.sh dense   test_books   --collection books-fresh
#   ./ingest.sh dense   downloads    --collection papers-fresh --limit 5
#   ./ingest.sh sparse  books-fresh
#   ./ingest.sh sparse  papers-fresh
#   ./ingest.sh full    test_books   --collection books-fresh
#   ./ingest.sh list
#   ./ingest.sh delete  books-fresh
#   ./ingest.sh search  "What is RAG?" --collection books-fresh

set -euo pipefail

VENV="$(dirname "$0")/.venv/bin/python"

if [[ ! -x "$VENV" ]]; then
    echo "ERROR: .venv not found at $VENV" >&2
    exit 1
fi

CMD="${1:-help}"
shift || true

case "$CMD" in
    health)
        "$VENV" -m src.cli.main health "$@"
        ;;
    create)
        "$VENV" -m src.cli.main create-collection "$@"
        ;;
    dense)
        "$VENV" -m src.cli.main ingest-dense "$@"
        ;;
    sparse)
        "$VENV" -m src.cli.main ingest-sparse "$@"
        ;;
    full)
        "$VENV" -m src.cli.main ingest "$@"
        ;;
    list)
        "$VENV" -m src.cli.main list-collections "$@"
        ;;
    delete)
        "$VENV" -m src.cli.main delete-collection "$@"
        ;;
    search)
        "$VENV" -m src.cli.main search "$@"
        ;;
    books)
        "$VENV" -m src.cli.main list-books "$@"
        ;;
    help|--help|-h)
        echo "Usage: ./ingest.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  health                          Check embedding server"
        echo "  create  <collection>            Create named-vector collection"
        echo "  dense   <dir> --collection <c>  Pass 1: load + chunk + embed dense"
        echo "                [--tokenizer-json <path>]"
        echo "  sparse  <collection>            Pass 2: scroll + embed sparse"
        echo "  full    <dir> --collection <c>  All three steps in one shot"
        echo "                [--tokenizer-json <path>]"
        echo "  list                            List all collections"
        echo "  delete  <collection>            Delete a collection"
        echo "  search  <query> [--collection]  Search a collection"
        echo "  books   [--collection <c>]      List books in a collection"
        ;;
    *)
        echo "Unknown command: $CMD" >&2
        echo "Run ./ingest.sh help for usage." >&2
        exit 1
        ;;
esac
