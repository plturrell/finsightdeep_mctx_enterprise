#!/usr/bin/env python3
"""
Real connection test script for SAP HANA and DataSphere.

This script verifies that connections to both systems are working
properly using the actual credentials.
"""

import sys
import os
import logging
import json
from datetime import datetime
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("connection_test")

# Set working directory to project root
os.chdir(os.path.join(os.path.dirname(__file__), "../../../"))

# Add the api directory to the Python path for imports
sys.path.append(os.path.abspath("api"))
logger.info(f"Python path: {sys.path}")

# Import the modules
try:
    from app.core.secrets import HanaSecrets, DataSphereSecrets, initialize_secrets
    from app.db.hana_connector import hana_manager
    from app.db.datasphere_connector import datasphere
    
    logger.info("Successfully imported all modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {str(e)}")
    sys.exit(1)


def test_hana_connection():
    """Test connection to SAP HANA database."""
    logger.info("Testing SAP HANA database connection...")
    
    # We'll try multiple connection approaches
    from hdbcli import dbapi
    
    # Get credentials
    connection_params = HanaSecrets.get_db_connection_params()
    logger.info(f"Got connection parameters for host: {connection_params['host']}")
    
    # Try approach 1: Standard connection parameters
    try:
        logger.info("Approach 1: Standard connection parameters")
        hana_params = {
            "address": connection_params["host"],
            "port": connection_params["port"],
            "user": connection_params["user"],
            "password": connection_params["password"],
            "encrypt": True,
            "sslValidateCertificate": True,
            "timeout": 30,
        }
        
        logger.info("Attempting direct connection to SAP HANA...")
        conn = dbapi.connect(**hana_params)
        
        if conn:
            logger.info("Successfully connected to SAP HANA")
            
            # Try a simple query
            with conn.cursor() as cursor:
                cursor.execute("SELECT 'HANA connection successful' as message FROM DUMMY")
                result = cursor.fetchone()
                
                if result and result[0] == "HANA connection successful":
                    logger.info("✅ SAP HANA connection test: SUCCESSFUL")
                    # Close the connection
                    conn.close()
                    return True
                else:
                    logger.error(f"❌ SAP HANA connection test: FAILED - Unexpected result: {result}")
                    conn.close()
            
        logger.info("Approach 1 failed, trying alternative approaches")
    except Exception as e:
        logger.error(f"Approach 1 failed: {str(e)}")
    
    # Try approach 2: Cloud authentication approach
    try:
        logger.info("Approach 2: Cloud authentication approach")
        cloud_params = {
            "address": connection_params["host"],
            "port": connection_params["port"], 
            "user": HanaSecrets.CLOUD_CLIENT_KEY,
            "password": HanaSecrets.CLOUD_CLIENT_SECRET,
            "encrypt": True,
            "sslValidateCertificate": True,
            "timeout": 30,
            "connectWithCloud": True,
        }
        
        logger.info("Attempting cloud connection to SAP HANA...")
        conn = dbapi.connect(**cloud_params)
        
        if conn:
            logger.info("Successfully connected to SAP HANA with cloud approach")
            
            # Try a simple query
            with conn.cursor() as cursor:
                cursor.execute("SELECT 'HANA connection successful' as message FROM DUMMY")
                result = cursor.fetchone()
                
                if result and result[0] == "HANA connection successful":
                    logger.info("✅ SAP HANA connection test: SUCCESSFUL")
                    # Close the connection
                    conn.close()
                    return True
                else:
                    logger.error(f"❌ SAP HANA connection test: FAILED - Unexpected result: {result}")
                    conn.close()
            
        logger.info("Approach 2 failed, trying alternative approaches")
    except Exception as e:
        logger.error(f"Approach 2 failed: {str(e)}")
    
    # Try approach 3: Connection string approach
    try:
        logger.info("Approach 3: Connection string approach")
        
        # Connection string approach
        conn_str = f"SERVERNODE={connection_params['host']}:{connection_params['port']};UID={connection_params['user']};PWD={connection_params['password']};ENCRYPT=TRUE;sslValidateCertificate=false"
        logger.info(f"Using connection string: {conn_str}")
        
        conn = dbapi.connect(conn_str)
        
        if conn:
            logger.info("Successfully connected to SAP HANA with connection string")
            
            # Try a simple query
            with conn.cursor() as cursor:
                cursor.execute("SELECT 'HANA connection successful' as message FROM DUMMY")
                result = cursor.fetchone()
                
                if result and result[0] == "HANA connection successful":
                    logger.info("✅ SAP HANA connection test: SUCCESSFUL")
                    # Close the connection
                    conn.close()
                    return True
                else:
                    logger.error(f"❌ SAP HANA connection test: FAILED - Unexpected result: {result}")
                    conn.close()
        
        logger.info("All connection approaches failed")
        return False
            
    except Exception as e:
        logger.error(f"All connection approaches failed: {str(e)}")
        return False


def test_datasphere_connection():
    """Test connection to SAP DataSphere."""
    logger.info("Testing SAP DataSphere connection...")
    
    try:
        # Get the credentials directly
        credentials = DataSphereSecrets.get_client_credentials()
        auth_urls = DataSphereSecrets.get_oauth_urls()
        api_url = DataSphereSecrets.get_api_url()
        
        logger.info(f"DataSphere client ID: {credentials['client_id']}")
        logger.info(f"DataSphere token URL: {auth_urls['token_url']}")
        logger.info(f"DataSphere API URL: {api_url}")
        
        # Try to get token directly
        import requests
        
        logger.info("Attempting to get token directly...")
        try:
            response = requests.post(
                auth_urls["token_url"],
                data={
                    "grant_type": "client_credentials",
                    "client_id": credentials["client_id"],
                    "client_secret": credentials["client_secret"],
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data["access_token"]
                logger.info("Successfully obtained DataSphere access token directly")
                
                # Try to get spaces with the token
                spaces_response = requests.get(
                    f"{api_url}/api/v1/dwc/catalog/spaces",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }
                )
                
                if spaces_response.status_code == 200:
                    spaces_data = spaces_response.json()
                    spaces = spaces_data.get("value", [])
                    
                    logger.info(f"✅ DataSphere connection test: SUCCESSFUL - Found {len(spaces)} spaces")
                    
                    # Print space info
                    for space in spaces:
                        logger.info(f"Space: {space.get('name')} (ID: {space.get('id')})")
                    
                    return True
                else:
                    logger.error(f"❌ DataSphere spaces API call failed: {spaces_response.status_code} {spaces_response.text}")
                    return False
            else:
                logger.error(f"❌ DataSphere token request failed: {response.status_code} {response.text}")
                return False
                
        except Exception as token_err:
            logger.error(f"❌ Error getting token directly: {str(token_err)}")
            
        # Fall back to using the connector
        logger.info("Falling back to connector method...")
        datasphere.initialize()
        logger.info("DataSphere connector initialized")
        
        # Get an access token
        access_token = datasphere.auth.get_access_token()
        if not access_token:
            logger.error("❌ DataSphere connection test: FAILED - Could not obtain access token")
            return False
            
        logger.info("Successfully obtained DataSphere access token")
            
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
    
    # Track success status
    hana_success = False
    datasphere_success = False
    
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
    try:
        # Get connection parameters directly from secrets
        connection_params = HanaSecrets.get_db_connection_params()
        hana_params = {
            "address": connection_params["host"],
            "port": connection_params["port"],
            "user": connection_params["user"],
            "password": connection_params["password"],
            "encrypt": True,
            "sslValidateCertificate": True,
            "timeout": 30,
        }
        
        # Connect directly
        from hdbcli import dbapi
        import json
        
        conn = dbapi.connect(**hana_params)
        
        if conn:
            logger.info("Connected to HANA for data persistence test")
            
            # Check if MCTX_SEARCH_HISTORY table exists
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM SYS.TABLES WHERE TABLE_NAME = 'MCTX_SEARCH_HISTORY'")
                result = cursor.fetchone()
                
                if result[0] == 0:
                    logger.info("Creating MCTX_SEARCH_HISTORY table...")
                    cursor.execute("""
                    CREATE TABLE MCTX_SEARCH_HISTORY (
                        ID BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        USER_ID VARCHAR(100),
                        SEARCH_TYPE VARCHAR(50) NOT NULL,
                        BATCH_SIZE INTEGER NOT NULL,
                        NUM_SIMULATIONS INTEGER NOT NULL,
                        MAX_DEPTH INTEGER,
                        CONFIG NCLOB,
                        DURATION_MS DECIMAL(10,2),
                        NUM_EXPANDED_NODES INTEGER,
                        MAX_DEPTH_REACHED INTEGER,
                        RESULT NCLOB
                    )
                    """)
                    conn.commit()
                    logger.info("MCTX_SEARCH_HISTORY table created")
            
            # Insert test data
            with conn.cursor() as cursor:
                cursor.execute("""
                INSERT INTO MCTX_SEARCH_HISTORY (
                    USER_ID, SEARCH_TYPE, BATCH_SIZE, NUM_SIMULATIONS, MAX_DEPTH,
                    CONFIG, DURATION_MS, NUM_EXPANDED_NODES, MAX_DEPTH_REACHED, RESULT
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    test_data["user_id"], 
                    test_data["search_type"], 
                    test_data["batch_size"], 
                    test_data["num_simulations"], 
                    test_data["max_depth"],
                    json.dumps(test_data["config"]), 
                    test_data["duration_ms"], 
                    test_data["num_expanded_nodes"], 
                    test_data["max_depth_reached"],
                    json.dumps(test_data["result"])
                ))
                
                # Get the inserted ID
                cursor.execute("SELECT CURRENT_IDENTITY_VALUE() FROM DUMMY")
                record_id = cursor.fetchone()[0]
                
                conn.commit()
                logger.info(f"✅ HANA data persistence test: SUCCESSFUL - Record ID: {record_id}")
                hana_success = True
                
            # Close the connection
            conn.close()
        else:
            logger.error("❌ HANA data persistence test: FAILED - Could not connect to database")
    except Exception as e:
        logger.error(f"❌ HANA data persistence test: FAILED - {str(e)}")
    
    # Test saving to DataSphere
    logger.info("Testing saving to DataSphere...")
    try:
        # Get credentials and token
        credentials = DataSphereSecrets.get_client_credentials()
        auth_urls = DataSphereSecrets.get_oauth_urls()
        api_url = DataSphereSecrets.get_api_url()
        space_id = DataSphereSecrets.get_space_id()
        
        import requests
        import json
        
        # Get token - try different formats for auth
        try:
            # First try: Basic auth
            logger.info("Trying with Basic Auth...")
            import base64
            auth_string = f"{credentials['client_id']}:{credentials['client_secret']}"
            encoded_auth = base64.b64encode(auth_string.encode()).decode()
            
            response = requests.post(
                auth_urls["token_url"],
                data={"grant_type": "client_credentials"},
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": f"Basic {encoded_auth}"
                }
            )
            
            if response.status_code != 200:
                # Second try: Form data
                logger.info("Trying with form data...")
                response = requests.post(
                    auth_urls["token_url"],
                    data={
                        "grant_type": "client_credentials",
                        "client_id": credentials["client_id"],
                        "client_secret": credentials["client_secret"],
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code != 200:
                    # Third try: Direct URL formatting
                    logger.info("Trying with direct URL format...")
                    url_with_auth = f"{auth_urls['token_url']}?grant_type=client_credentials&client_id={credentials['client_id']}&client_secret={credentials['client_secret']}"
                    response = requests.post(url_with_auth)
        except Exception as auth_err:
            logger.error(f"Error during authentication attempts: {str(auth_err)}")
            response = requests.post(
                auth_urls["token_url"],
                data={
                    "grant_type": "client_credentials",
                    "client_id": credentials["client_id"],
                    "client_secret": credentials["client_secret"],
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data["access_token"]
            
            # Prepare data record
            record = {
                "searchType": test_data["search_type"],
                "batchSize": test_data["batch_size"],
                "numSimulations": test_data["num_simulations"],
                "durationMs": test_data["duration_ms"],
                "numExpandedNodes": test_data["num_expanded_nodes"],
                "timestamp": test_data["timestamp"],
                "userId": test_data["user_id"] or "anonymous",
                "results": json.dumps(test_data["result"]),
            }
            
            # Try to save to DataSphere
            try:
                save_response = requests.post(
                    f"{api_url}/api/v1/dwc/catalog/spaces('{space_id}')/tables('MCTS_SEARCH_HISTORY')/data",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json={"data": [record]}
                )
                
                if save_response.status_code in [200, 201, 202, 204]:
                    logger.info(f"✅ DataSphere data persistence test: SUCCESSFUL")
                    datasphere_success = True
                else:
                    logger.error(f"❌ DataSphere data persistence test: API error - {save_response.status_code} {save_response.text}")
            except Exception as save_err:
                logger.error(f"❌ DataSphere data persistence test: Save error - {str(save_err)}")
        else:
            logger.error(f"❌ DataSphere data persistence test: Token error - {response.status_code} {response.text}")
    except Exception as ds_err:
        logger.error(f"❌ DataSphere data persistence test: FAILED - {str(ds_err)}")
    
    # Return overall success
    return hana_success or datasphere_success


def main():
    """Main test function."""
    logger.info("Starting real connection tests...")
    
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