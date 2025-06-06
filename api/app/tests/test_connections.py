#!/usr/bin/env python3
"""
Connection test script for SAP HANA and DataSphere.

This script verifies that connections to both systems are working
properly using the actual credentials.
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add parent directory to path to allow importing from app modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("connection_test")

# Import connection modules
from db.hana_connector import hana_manager
from db.datasphere_connector import datasphere
from core.secrets import HanaSecrets, DataSphereSecrets, initialize_secrets


def test_hana_connection():
    """Test connection to SAP HANA database."""
    logger.info("Testing SAP HANA database connection...")
    
    try:
        # Initialize HANA connection manager
        hana_manager.initialize()
        
        # Try a simple query
        with hana_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 'HANA connection successful' as message FROM DUMMY")
                result = cursor.fetchone()
                
                if result and result[0] == "HANA connection successful":
                    logger.info("✅ SAP HANA connection test: SUCCESSFUL")
                    return True
                else:
                    logger.error("❌ SAP HANA connection test: FAILED - Unexpected result")
                    return False
    except Exception as e:
        logger.error(f"❌ SAP HANA connection test: FAILED - {str(e)}")
        return False


def test_datasphere_connection():
    """Test connection to SAP DataSphere."""
    logger.info("Testing SAP DataSphere connection...")
    
    try:
        # Initialize DataSphere connector
        datasphere.initialize()
        
        # Get an access token
        access_token = datasphere.auth.get_access_token()
        if not access_token:
            logger.error("❌ DataSphere connection test: FAILED - Could not obtain access token")
            return False
            
        # Get list of spaces
        spaces = datasphere.get_spaces()
        if spaces is not None:
            logger.info(f"✅ DataSphere connection test: SUCCESSFUL - Found {len(spaces)} spaces")
            
            # Print space info
            for space in spaces:
                logger.info(f"Space: {space.get('name')} (ID: {space.get('id')})")
            
            return True
        else:
            logger.error("❌ DataSphere connection test: FAILED - Could not retrieve spaces")
            return False
    except Exception as e:
        logger.error(f"❌ DataSphere connection test: FAILED - {str(e)}")
        return False


def test_save_data():
    """Test saving data to both systems."""
    logger.info("Testing data persistence...")
    
    try:
        # Test data
        test_data = {
            "search_type": "test_search",
            "batch_size": 2,
            "num_simulations": 10,
            "max_depth": 5,
            "duration_ms": 123.45,
            "num_expanded_nodes": 20,
            "max_depth_reached": 3,
            "timestamp": datetime.now().isoformat(),
            "user_id": "test_user",
            "config": {"test": True},
            "result": {"action": [0, 1], "action_weights": [[0.7, 0.3], [0.4, 0.6]]}
        }
        
        # Test saving to HANA
        logger.info("Testing saving to SAP HANA...")
        record_id = hana_manager.save_search_history(
            search_type=test_data["search_type"],
            batch_size=test_data["batch_size"],
            num_simulations=test_data["num_simulations"],
            max_depth=test_data["max_depth"],
            config=test_data["config"],
            duration_ms=test_data["duration_ms"],
            num_expanded_nodes=test_data["num_expanded_nodes"],
            max_depth_reached=test_data["max_depth_reached"],
            result=test_data["result"],
            user_id=test_data["user_id"]
        )
        
        logger.info(f"✅ HANA data persistence test: SUCCESSFUL - Record ID: {record_id}")
        
        # Test saving to DataSphere
        logger.info("Testing saving to DataSphere...")
        try:
            result = datasphere.save_search_results(
                search_type=test_data["search_type"],
                batch_size=test_data["batch_size"],
                num_simulations=test_data["num_simulations"],
                duration_ms=test_data["duration_ms"],
                num_expanded_nodes=test_data["num_expanded_nodes"],
                results=test_data["result"],
                user_id=test_data["user_id"]
            )
            
            if result:
                logger.info(f"✅ DataSphere data persistence test: SUCCESSFUL")
            else:
                logger.warning("⚠️ DataSphere data persistence test: No result returned")
                
        except Exception as ds_err:
            logger.error(f"❌ DataSphere data persistence test: FAILED - {str(ds_err)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Data persistence test: FAILED - {str(e)}")
        return False


def main():
    """Main test function."""
    logger.info("Starting connection tests...")
    
    # Initialize secrets
    try:
        initialize_secrets()
        logger.info("✅ Secrets initialization: SUCCESSFUL")
    except Exception as e:
        logger.error(f"❌ Secrets initialization: FAILED - {str(e)}")
        return 1
    
    # Test HANA connection
    hana_success = test_hana_connection()
    
    # Test DataSphere connection
    datasphere_success = test_datasphere_connection()
    
    # Test data persistence if connections are successful
    if hana_success and datasphere_success:
        data_success = test_save_data()
    else:
        data_success = False
        logger.warning("⚠️ Skipping data persistence tests due to connection failures")
    
    # Report results
    logger.info("Connection Test Results:")
    logger.info(f"SAP HANA Connection: {'✅ PASS' if hana_success else '❌ FAIL'}")
    logger.info(f"DataSphere Connection: {'✅ PASS' if datasphere_success else '❌ FAIL'}")
    logger.info(f"Data Persistence: {'✅ PASS' if data_success else '❌ FAIL' if hana_success and datasphere_success else '⚠️ SKIPPED'}")
    
    # Return success if all tests pass
    return 0 if (hana_success and datasphere_success and (data_success or not (hana_success and datasphere_success))) else 1


if __name__ == "__main__":
    sys.exit(main())