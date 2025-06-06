#!/usr/bin/env python3
"""
Simple test script for SAP HANA connection.

This script focuses only on testing the SAP HANA connection
with the provided credentials.
"""

import sys
import os
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hana_test")

# Set working directory to project root
os.chdir(os.path.join(os.path.dirname(__file__), "../../../"))

# Add the api directory to the Python path for imports
sys.path.append(os.path.abspath("api"))
logger.info(f"Python path: {sys.path}")

try:
    from app.core.secrets import HanaSecrets
    from hdbcli import dbapi
    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {str(e)}")
    logger.info("Installing hdbcli package...")
    try:
        import pip
        pip.main(['install', '--user', 'hdbcli'])
        from hdbcli import dbapi
        logger.info("Successfully installed hdbcli")
    except Exception as pip_error:
        logger.error(f"Failed to install hdbcli: {str(pip_error)}")
        sys.exit(1)


def test_hana_connection():
    """Test direct connection to SAP HANA."""
    logger.info("Testing direct SAP HANA connection...")
    
    # Get connection parameters
    connection_params = HanaSecrets.get_db_connection_params()
    logger.info(f"Using connection parameters:")
    logger.info(f"  Host: {connection_params['host']}")
    logger.info(f"  Port: {connection_params['port']}")
    logger.info(f"  User: {connection_params['user']}")
    # Don't log the password
    
    # Try multiple connection approaches
    try_direct_connection(connection_params)
    try_ssl_connection(connection_params)
    try_dsn_connection(connection_params)


def try_direct_connection(params):
    """Try connecting with direct parameters."""
    logger.info("\nAttempt 1: Direct connection parameters")
    try:
        # Map parameters to hdbcli format
        conn_params = {
            "address": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "encrypt": True,
            "sslValidateCertificate": False,
            "connectTimeout": 30000,
        }
        
        logger.info("Connecting to SAP HANA...")
        start_time = time.time()
        conn = dbapi.connect(**conn_params)
        
        if conn:
            end_time = time.time()
            logger.info(f"✅ Connection established in {(end_time - start_time)*1000:.2f} ms")
            
            # Test a simple query
            logger.info("Executing test query...")
            with conn.cursor() as cursor:
                cursor.execute("SELECT 'HANA connection successful' as message FROM DUMMY")
                result = cursor.fetchone()
                logger.info(f"Query result: {result}")
            
            # Close the connection
            conn.close()
            logger.info("Connection closed")
        else:
            logger.error("❌ Connection failed")
    except Exception as e:
        logger.error(f"❌ Connection error: {str(e)}")


def try_ssl_connection(params):
    """Try connecting with SSL parameters."""
    logger.info("\nAttempt 2: SSL connection parameters")
    try:
        # Map parameters to hdbcli format with SSL options
        conn_params = {
            "address": params["host"],
            "port": params["port"],
            "user": params["user"],
            "password": params["password"],
            "encrypt": True,
            "sslValidateCertificate": False,
            "sslHostNameInCertificate": "*",
            "connectTimeout": 30000,
        }
        
        logger.info("Connecting to SAP HANA with SSL...")
        start_time = time.time()
        conn = dbapi.connect(**conn_params)
        
        if conn:
            end_time = time.time()
            logger.info(f"✅ SSL Connection established in {(end_time - start_time)*1000:.2f} ms")
            
            # Test a simple query
            logger.info("Executing test query...")
            with conn.cursor() as cursor:
                cursor.execute("SELECT 'HANA connection successful' as message FROM DUMMY")
                result = cursor.fetchone()
                logger.info(f"Query result: {result}")
            
            # Close the connection
            conn.close()
            logger.info("Connection closed")
        else:
            logger.error("❌ SSL Connection failed")
    except Exception as e:
        logger.error(f"❌ SSL Connection error: {str(e)}")


def try_dsn_connection(params):
    """Try connecting with connection string."""
    logger.info("\nAttempt 3: Connection string (DSN)")
    try:
        # Build connection string
        conn_string = (
            f"SERVERNODE={params['host']}:{params['port']};"
            f"UID={params['user']};"
            f"PWD={params['password']};"
            "ENCRYPT=TRUE;"
            "sslValidateCertificate=false"
        )
        
        logger.info(f"Using connection string: {conn_string.replace(params['password'], '******')}")
        
        start_time = time.time()
        conn = dbapi.connect(conn_string)
        
        if conn:
            end_time = time.time()
            logger.info(f"✅ DSN Connection established in {(end_time - start_time)*1000:.2f} ms")
            
            # Test a simple query
            logger.info("Executing test query...")
            with conn.cursor() as cursor:
                cursor.execute("SELECT 'HANA connection successful' as message FROM DUMMY")
                result = cursor.fetchone()
                logger.info(f"Query result: {result}")
            
            # Get database information
            logger.info("Getting database information:")
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM M_DATABASE")
                result = cursor.fetchone()
                if result:
                    logger.info(f"  Database name: {result[0]}")
                    logger.info(f"  Database version: {result[1]}")
                
                cursor.execute("SELECT COUNT(*) FROM TABLES")
                count = cursor.fetchone()[0]
                logger.info(f"  Number of tables: {count}")
            
            # Close the connection
            conn.close()
            logger.info("Connection closed")
        else:
            logger.error("❌ DSN Connection failed")
    except Exception as e:
        logger.error(f"❌ DSN Connection error: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting SAP HANA connection test...")
    test_hana_connection()
    logger.info("SAP HANA connection test completed")