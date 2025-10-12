"""
Database utilities for managing DuckDB databases for asteroid observations.

This module provides functions to initialize and manage separate DuckDB databases
for MPC, Miriade, and ZTF observation data, as well as a metadata database to track
what data has been downloaded and stored.
"""

import duckdb
import pandas as pd
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger('mpc_viewer')

# Database paths
DB_DIR = './db/'
MPC_DB_PATH = os.path.join(DB_DIR, 'mpc.duckdb')
MIRIADE_DB_PATH = os.path.join(DB_DIR, 'miriade.duckdb')
ZTF_DB_PATH = os.path.join(DB_DIR, 'ztf.duckdb')
METADATA_DB_PATH = os.path.join(DB_DIR, 'metadata.duckdb')


def ensure_db_directory():
    """Ensure the database directory exists."""
    os.makedirs(DB_DIR, exist_ok=True)
    logger.info(f"Ensured database directory exists: {DB_DIR}")


def init_metadata_db():
    """
    Initialize the metadata database with tables to track downloaded data.
    
    The metadata database contains:
    - asteroids: Basic information about asteroids
    - data_downloads: Records of when data was downloaded for each asteroid
    """
    ensure_db_directory()
    
    conn = duckdb.connect(METADATA_DB_PATH)
    
    # Create asteroids table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS asteroids (
            asteroid_number VARCHAR PRIMARY KEY,
            iau_designation VARCHAR,
            safe_designation VARCHAR,
            iau_name VARCHAR,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create data_downloads table to track what data has been fetched
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_downloads (
            id INTEGER PRIMARY KEY,
            asteroid_number VARCHAR,
            data_source VARCHAR,  -- 'mpc', 'miriade', 'ztf'
            download_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            record_count INTEGER,
            status VARCHAR,  -- 'success', 'error', 'partial'
            error_message VARCHAR,
            FOREIGN KEY (asteroid_number) REFERENCES asteroids(asteroid_number)
        )
    """)
    
    # Create sequence for data_downloads id
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS data_downloads_seq START 1
    """)
    
    conn.close()
    logger.info("Metadata database initialized")


def init_mpc_db():
    """
    Initialize the MPC database.
    Each asteroid will have its own table named by safe_designation.
    """
    ensure_db_directory()
    conn = duckdb.connect(MPC_DB_PATH)
    conn.close()
    logger.info("MPC database initialized")


def init_miriade_db():
    """
    Initialize the Miriade database.
    Each asteroid will have its own table named by safe_designation.
    """
    ensure_db_directory()
    conn = duckdb.connect(MIRIADE_DB_PATH)
    conn.close()
    logger.info("Miriade database initialized")


def init_ztf_db():
    """
    Initialize the ZTF database.
    Each asteroid will have its own table named by safe_designation.
    """
    ensure_db_directory()
    conn = duckdb.connect(ZTF_DB_PATH)
    conn.close()
    logger.info("ZTF database initialized")


def init_all_databases():
    """Initialize all databases."""
    init_metadata_db()
    init_mpc_db()
    init_miriade_db()
    init_ztf_db()
    logger.info("All databases initialized")


def register_asteroid(asteroid_number: str, iau_designation: str, 
                     safe_designation: str, iau_name: Optional[str] = None):
    """
    Register an asteroid in the metadata database.
    
    Args:
        asteroid_number: The asteroid number or temporary designation
        iau_designation: The IAU designation
        safe_designation: Filesystem-safe designation
        iau_name: Optional name of the asteroid
    """
    conn = duckdb.connect(METADATA_DB_PATH)
    
    # Check if asteroid already exists
    result = conn.execute(
        "SELECT asteroid_number FROM asteroids WHERE asteroid_number = ?",
        [asteroid_number]
    ).fetchone()
    
    if result is None:
        # Insert new asteroid
        conn.execute("""
            INSERT INTO asteroids (asteroid_number, iau_designation, safe_designation, iau_name)
            VALUES (?, ?, ?, ?)
        """, [asteroid_number, iau_designation, safe_designation, iau_name])
        logger.info(f"Registered new asteroid: {asteroid_number} ({iau_designation})")
    else:
        # Update existing asteroid
        conn.execute("""
            UPDATE asteroids 
            SET iau_designation = ?, safe_designation = ?, iau_name = ?, 
                last_updated = CURRENT_TIMESTAMP
            WHERE asteroid_number = ?
        """, [iau_designation, safe_designation, iau_name, asteroid_number])
        logger.info(f"Updated asteroid record: {asteroid_number}")
    
    conn.close()


def get_asteroid_info(asteroid_number: str) -> Optional[Dict[str, Any]]:
    """
    Get asteroid information from metadata database.
    
    Args:
        asteroid_number: The asteroid number or designation
        
    Returns:
        Dictionary with asteroid information or None if not found
    """
    conn = duckdb.connect(METADATA_DB_PATH)
    
    result = conn.execute("""
        SELECT asteroid_number, iau_designation, safe_designation, iau_name, 
               first_seen, last_updated
        FROM asteroids 
        WHERE asteroid_number = ?
    """, [asteroid_number]).fetchone()
    
    conn.close()
    
    if result:
        return {
            'asteroid_number': result[0],
            'iau_designation': result[1],
            'safe_designation': result[2],
            'iau_name': result[3],
            'first_seen': result[4],
            'last_updated': result[5]
        }
    return None


def record_data_download(asteroid_number: str, data_source: str, 
                        record_count: int, status: str = 'success',
                        error_message: Optional[str] = None):
    """
    Record a data download event in the metadata database.
    
    Args:
        asteroid_number: The asteroid number
        data_source: Source of data ('mpc', 'miriade', 'ztf')
        record_count: Number of records downloaded
        status: Status of download ('success', 'error', 'partial')
        error_message: Optional error message if status is error
    """
    conn = duckdb.connect(METADATA_DB_PATH)
    
    conn.execute("""
        INSERT INTO data_downloads 
        (id, asteroid_number, data_source, record_count, status, error_message)
        VALUES (nextval('data_downloads_seq'), ?, ?, ?, ?, ?)
    """, [asteroid_number, data_source, record_count, status, error_message])
    
    conn.close()
    logger.info(f"Recorded {data_source} download for {asteroid_number}: {record_count} records, status: {status}")


def save_mpc_data(safe_designation: str, df: pd.DataFrame):
    """
    Save MPC observation data to the database.
    
    Args:
        safe_designation: Safe asteroid designation (table name)
        df: DataFrame with MPC observation data
    """
    conn = duckdb.connect(MPC_DB_PATH)
    
    table_name = f"mpc_{safe_designation}"
    
    # Drop existing table if it exists
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Create table from DataFrame
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    
    conn.close()
    logger.info(f"Saved {len(df)} MPC observations for {safe_designation}")


def load_mpc_data(safe_designation: str) -> Optional[pd.DataFrame]:
    """
    Load MPC observation data from the database.
    
    Args:
        safe_designation: Safe asteroid designation (table name)
        
    Returns:
        DataFrame with MPC data or None if not found
    """
    conn = duckdb.connect(MPC_DB_PATH)
    
    table_name = f"mpc_{safe_designation}"
    
    try:
        df = conn.execute(f"SELECT * FROM {table_name}").df()
        conn.close()
        logger.info(f"Loaded {len(df)} MPC observations for {safe_designation}")
        return df
    except Exception as e:
        conn.close()
        logger.debug(f"No MPC data found for {safe_designation}: {str(e)}")
        return None


def mpc_data_exists(safe_designation: str) -> bool:
    """
    Check if MPC data exists for an asteroid.
    
    Args:
        safe_designation: Safe asteroid designation
        
    Returns:
        True if data exists, False otherwise
    """
    conn = duckdb.connect(MPC_DB_PATH)
    
    table_name = f"mpc_{safe_designation}"
    
    try:
        result = conn.execute(f"""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()
        conn.close()
        return result is not None
    except Exception:
        conn.close()
        return False


def save_miriade_data(safe_designation: str, df: pd.DataFrame):
    """
    Save Miriade ephemeris data to the database.
    
    Args:
        safe_designation: Safe asteroid designation (table name)
        df: DataFrame with Miriade data
    """
    conn = duckdb.connect(MIRIADE_DB_PATH)
    
    table_name = f"miriade_{safe_designation}"
    
    # Drop existing table if it exists
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Create table from DataFrame
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    
    conn.close()
    logger.info(f"Saved {len(df)} Miriade records for {safe_designation}")


def load_miriade_data(safe_designation: str) -> Optional[pd.DataFrame]:
    """
    Load Miriade ephemeris data from the database.
    
    Args:
        safe_designation: Safe asteroid designation (table name)
        
    Returns:
        DataFrame with Miriade data or None if not found
    """
    conn = duckdb.connect(MIRIADE_DB_PATH)
    
    table_name = f"miriade_{safe_designation}"
    
    try:
        df = conn.execute(f"SELECT * FROM {table_name}").df()
        conn.close()
        logger.info(f"Loaded {len(df)} Miriade records for {safe_designation}")
        return df
    except Exception as e:
        conn.close()
        logger.debug(f"No Miriade data found for {safe_designation}: {str(e)}")
        return None


def miriade_data_exists(safe_designation: str) -> bool:
    """
    Check if Miriade data exists for an asteroid.
    
    Args:
        safe_designation: Safe asteroid designation
        
    Returns:
        True if data exists, False otherwise
    """
    conn = duckdb.connect(MIRIADE_DB_PATH)
    
    table_name = f"miriade_{safe_designation}"
    
    try:
        result = conn.execute(f"""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()
        conn.close()
        return result is not None
    except Exception:
        conn.close()
        return False


def save_ztf_data(safe_designation: str, df: pd.DataFrame):
    """
    Save ZTF observation data to the database.
    
    Args:
        safe_designation: Safe asteroid designation (table name)
        df: DataFrame with ZTF data
    """
    conn = duckdb.connect(ZTF_DB_PATH)
    
    table_name = f"ztf_{safe_designation}"
    
    # Drop existing table if it exists
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Create table from DataFrame
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    
    conn.close()
    logger.info(f"Saved {len(df)} ZTF observations for {safe_designation}")


def load_ztf_data(safe_designation: str) -> Optional[pd.DataFrame]:
    """
    Load ZTF observation data from the database.
    
    Args:
        safe_designation: Safe asteroid designation (table name)
        
    Returns:
        DataFrame with ZTF data or None if not found
    """
    conn = duckdb.connect(ZTF_DB_PATH)
    
    table_name = f"ztf_{safe_designation}"
    
    try:
        df = conn.execute(f"SELECT * FROM {table_name}").df()
        conn.close()
        logger.info(f"Loaded {len(df)} ZTF observations for {safe_designation}")
        return df
    except Exception as e:
        conn.close()
        logger.debug(f"No ZTF data found for {safe_designation}: {str(e)}")
        return None


def ztf_data_exists(safe_designation: str) -> bool:
    """
    Check if ZTF data exists for an asteroid.
    
    Args:
        safe_designation: Safe asteroid designation
        
    Returns:
        True if data exists, False otherwise
    """
    conn = duckdb.connect(ZTF_DB_PATH)
    
    table_name = f"ztf_{safe_designation}"
    
    try:
        result = conn.execute(f"""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()
        conn.close()
        return result is not None
    except Exception:
        conn.close()
        return False


def get_download_history(asteroid_number: str) -> pd.DataFrame:
    """
    Get download history for an asteroid.
    
    Args:
        asteroid_number: The asteroid number
        
    Returns:
        DataFrame with download history
    """
    conn = duckdb.connect(METADATA_DB_PATH)
    
    df = conn.execute("""
        SELECT data_source, download_timestamp, record_count, status, error_message
        FROM data_downloads
        WHERE asteroid_number = ?
        ORDER BY download_timestamp DESC
    """, [asteroid_number]).df()
    
    conn.close()
    return df


def list_all_asteroids() -> pd.DataFrame:
    """
    List all asteroids in the metadata database.
    
    Returns:
        DataFrame with asteroid information
    """
    conn = duckdb.connect(METADATA_DB_PATH)
    
    df = conn.execute("""
        SELECT asteroid_number, iau_designation, safe_designation, iau_name, 
               first_seen, last_updated
        FROM asteroids
        ORDER BY last_updated DESC
    """).df()
    
    conn.close()
    return df
