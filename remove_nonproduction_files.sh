#!/bin/bash
# Script to remove non-production files from MCTX codebase

set -e

echo "Removing test-related files..."
rm -rf mctx/_src/tests/
rm -f test.sh

echo "Removing example files..."
rm -rf examples/

echo "Removing development requirement files..."
rm -f requirements/requirements-test.txt
rm -f requirements/requirements_examples.txt

echo "Removing contribution documentation..."
rm -f CONTRIBUTING.md

echo "Non-production files have been removed successfully."