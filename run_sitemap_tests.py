#!/usr/bin/env python3
"""
Test runner for sitemap parsing functionality.

This script runs the sitemap tests and provides detailed output about
what scenarios are being tested.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_tests(test_type="all", verbose=True):
    """
    Run sitemap tests.
    
    Args:
        test_type: Type of tests to run ("all", "unit", "integration", "scenarios")
        verbose: Whether to show verbose output
    """
    
    print("🧪 Running Sitemap Parsing Tests")
    print("=" * 50)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add specific test filters based on type
    if test_type == "unit":
        cmd.extend(["-m", "not integration", "tests/test_sitemap_parsing.py"])
        print("📋 Running Unit Tests Only")
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
        print("🔗 Running Integration Tests Only")
    elif test_type == "scenarios":
        cmd.append("tests/test_sitemap_changes_scenario.py")
        print("🎭 Running Scenario Tests Only")
    else:
        cmd.append("tests/")
        print("🎯 Running All Tests")
    
    print("=" * 50)
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend(["--cov=hubert.data_ingestion.huber_crawler", "--cov-report=term-missing"])
        print("📊 Coverage analysis enabled")
    except ImportError:
        print("ℹ️  Coverage analysis not available (install pytest-cov)")
    
    print()
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return False


def main():
    """Main function to handle command line arguments."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run sitemap parsing tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "scenarios"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed. Please install it with:")
        print("   pip install pytest")
        return 1
    
    # Check if tests directory exists
    tests_dir = project_root / "tests"
    if not tests_dir.exists():
        print(f"❌ Tests directory not found: {tests_dir}")
        return 1
    
    # Run the tests
    print(f"🚀 Starting sitemap tests in: {project_root}")
    
    success = run_tests(
        test_type=args.type,
        verbose=not args.quiet
    )
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 