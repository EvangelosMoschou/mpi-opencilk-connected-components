#!/usr/bin/env bash
# build_report.sh - compile Knn, NC Report.tex and clean auxiliary files
# Usage: ./build_report.sh [--remove-png]
# If --remove-png is passed the script will delete AUTH_logo.png after building.

set -euo pipefail
cd "$(dirname "$0")"

TEXFILE="report.tex"
OUTPDF="report.pdf"

echo "Compiling $TEXFILE..."
# Run pdflatex twice to resolve references; suppress terminal chatter
xelatex -interaction=nonstopmode -halt-on-error "$TEXFILE" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error "$TEXFILE" >/dev/null

echo "Compilation finished. Cleaning auxiliary files..."
# Remove common auxiliary files
rm -f "report.aux" "report.log" "report.out" "report.toc"

if [ "${1-}" = "--remove-png" ]; then
  echo "Removing AUTH_logo.png as requested..."
  rm -f AUTH_logo.png
fi

echo "Done. Generated: $OUTPDF"
