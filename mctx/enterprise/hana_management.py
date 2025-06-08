# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SAP HANA database management utilities for MCTX."""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mctx.enterprise.hana_management")

try:
    from mctx.enterprise.hana_integration import (
        HANA_AVAILABLE,
        HanaConfig,
        connect_to_hana,
    )
    from mctx.enterprise.enhanced_hana_integration import (
        clean_old_data,
        delete_simulation_results,
        delete_tree_from_hana,
        get_database_statistics,
        list_trees,
    )
except ImportError:
    logger.error("Failed to import HANA integration modules. "
                 "Make sure mctx is installed correctly.")
    HANA_AVAILABLE = False


def get_connection_from_env() -> Optional[Any]:
    """Create a HANA connection using environment variables.
    
    Returns:
        A HanaConnection object or None if connection fails.
    """
    if not HANA_AVAILABLE:
        logger.error("SAP HANA client libraries not available")
        return None
    
    try:
        # Get connection parameters from environment
        hana_host = os.environ.get("HANA_HOST")
        hana_port = int(os.environ.get("HANA_PORT", "443"))
        hana_user = os.environ.get("HANA_USER")
        hana_password = os.environ.get("HANA_PASSWORD")
        hana_schema = os.environ.get("HANA_SCHEMA", "MCTX")
        
        if not all([hana_host, hana_user, hana_password]):
            logger.error("Missing required HANA environment variables")
            return None
        
        # Create connection
        config = HanaConfig(
            host=hana_host,
            port=hana_port,
            user=hana_user,
            password=hana_password,
            schema=hana_schema,
        )
        
        connection = connect_to_hana(config)
        return connection
    except Exception as e:
        logger.error(f"Failed to create HANA connection: {e}")
        return None


def stats_command(args):
    """Show database statistics."""
    connection = get_connection_from_env()
    if not connection:
        return
    
    try:
        stats = get_database_statistics(connection)
        print(json.dumps(stats, indent=2, default=str))
    finally:
        connection.close_all()


def cleanup_command(args):
    """Clean up old data."""
    connection = get_connection_from_env()
    if not connection:
        return
    
    try:
        results = clean_old_data(
            connection, 
            older_than_days=args.days,
            simulation_results_only=not args.all
        )
        print(f"Cleanup complete: {results}")
    finally:
        connection.close_all()


def list_command(args):
    """List trees with filtering."""
    connection = get_connection_from_env()
    if not connection:
        return
    
    try:
        trees = list_trees(
            connection,
            name_filter=args.name,
            min_batch_size=args.min_batch,
            max_batch_size=args.max_batch,
            min_num_simulations=args.min_sims,
            max_num_simulations=args.max_sims,
            limit=args.limit,
            offset=args.offset
        )
        
        # Print formatted output
        if not trees:
            print("No trees found matching criteria")
            return
        
        print(f"Found {len(trees)} trees:")
        for tree in trees:
            print(f"ID: {tree['tree_id']}")
            print(f"  Name: {tree['name'] or 'Unnamed'}")
            print(f"  Batch size: {tree['batch_size']}")
            print(f"  Actions: {tree['num_actions']}")
            print(f"  Simulations: {tree['num_simulations']}")
            print(f"  Created: {tree['created_at']}")
            print(f"  Updated: {tree['updated_at']}")
            print()
    finally:
        connection.close_all()


def delete_command(args):
    """Delete trees or results."""
    connection = get_connection_from_env()
    if not connection:
        return
    
    try:
        if args.type == "tree":
            if args.id:
                # Delete specific tree
                success = delete_tree_from_hana(connection, args.id, args.cascade)
                print(f"Tree deletion {'successful' if success else 'failed'}")
            elif args.older_than:
                # Bulk delete old trees via cleanup
                older_than = datetime.datetime.now() - datetime.timedelta(days=args.older_than)
                results = clean_old_data(
                    connection,
                    older_than_days=args.older_than,
                    simulation_results_only=False
                )
                print(f"Deleted {results.get('trees', 0)} old trees")
        elif args.type == "results":
            if args.id:
                # Delete specific result
                count = delete_simulation_results(connection, result_id=args.id)
                print(f"Deleted {count} results")
            elif args.tree_id:
                # Delete results for a tree
                count = delete_simulation_results(connection, tree_id=args.tree_id)
                print(f"Deleted {count} results for tree {args.tree_id}")
            elif args.older_than:
                # Delete old results
                older_than = datetime.datetime.now() - datetime.timedelta(days=args.older_than)
                count = delete_simulation_results(connection, older_than=older_than)
                print(f"Deleted {count} results older than {args.older_than} days")
    finally:
        connection.close_all()


def main():
    """Main entry point for the command-line tool."""
    parser = argparse.ArgumentParser(description="SAP HANA Database Management for MCTX")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    
    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old data")
    cleanup_parser.add_argument("--days", type=int, default=30, help="Delete data older than this many days")
    cleanup_parser.add_argument("--all", action="store_true", help="Clean up all data types, not just simulation results")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List trees with filtering")
    list_parser.add_argument("--name", help="Filter by name (LIKE syntax)")
    list_parser.add_argument("--min-batch", type=int, help="Minimum batch size")
    list_parser.add_argument("--max-batch", type=int, help="Maximum batch size")
    list_parser.add_argument("--min-sims", type=int, help="Minimum number of simulations")
    list_parser.add_argument("--max-sims", type=int, help="Maximum number of simulations")
    list_parser.add_argument("--limit", type=int, default=100, help="Maximum number of results")
    list_parser.add_argument("--offset", type=int, default=0, help="Results offset")
    
    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete trees or results")
    delete_parser.add_argument("type", choices=["tree", "results"], help="Type of item to delete")
    delete_parser.add_argument("--id", help="ID of specific item to delete")
    delete_parser.add_argument("--tree-id", help="Tree ID (for deleting related results)")
    delete_parser.add_argument("--cascade", action="store_true", help="Also delete related items (for trees)")
    delete_parser.add_argument("--older-than", type=int, help="Delete items older than this many days")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if not HANA_AVAILABLE:
        print("Error: SAP HANA client libraries not available")
        print("Please install the required packages:")
        print("  pip install hdbcli==2.19.21")
        return
    
    # Execute the selected command
    if args.command == "stats":
        stats_command(args)
    elif args.command == "cleanup":
        cleanup_command(args)
    elif args.command == "list":
        list_command(args)
    elif args.command == "delete":
        delete_command(args)


if __name__ == "__main__":
    main()