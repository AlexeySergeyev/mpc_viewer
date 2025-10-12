"""
Migration script to convert existing CSV files to DuckDB databases.

This script reads CSV files from the old directory structure and imports them
into the new DuckDB databases.

Usage:
    python migrate_csv_to_duckdb.py
"""

import os
import pandas as pd
import glob
import logging
import db_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def extract_safe_designation(filename):
    """
    Extract safe designation from filename.
    
    Examples:
        './db/mpc/2000_LE29_mpc.csv.gz' -> '2000_LE29'
        './db/ztf/A807_FA_ztf.csv.gz' -> 'A807_FA'
    """
    basename = os.path.basename(filename)
    # Remove file extension
    name_without_ext = basename.replace('.csv.gz', '').replace('.csv', '')
    # Remove data source suffix
    for suffix in ['_mpc', '_ztf', '_miriade']:
        if name_without_ext.endswith(suffix):
            return name_without_ext[:-len(suffix)]
    return name_without_ext


def migrate_mpc_data():
    """Migrate MPC CSV files to DuckDB."""
    mpc_files = glob.glob('./db/mpc/*.csv.gz') + glob.glob('./db/mpc/*.csv')
    
    logger.info(f"Found {len(mpc_files)} MPC files to migrate")
    
    for filepath in mpc_files:
        try:
            safe_designation = extract_safe_designation(filepath)
            
            # Check if already in database
            if db_utils.mpc_data_exists(safe_designation):
                logger.info(f"MPC data for {safe_designation} already exists in database, skipping")
                continue
            
            # Read CSV file
            logger.info(f"Migrating MPC data: {filepath}")
            df = pd.read_csv(filepath)
            
            # Save to database
            db_utils.save_mpc_data(safe_designation, df)
            logger.info(f"Successfully migrated {len(df)} MPC records for {safe_designation}")
            
        except Exception as e:
            logger.error(f"Error migrating {filepath}: {str(e)}")


def migrate_miriade_data():
    """Migrate Miriade CSV files to DuckDB."""
    miriade_files = glob.glob('./db/miriade/*.csv.gz') + glob.glob('./db/miriade/*.csv')
    
    logger.info(f"Found {len(miriade_files)} Miriade files to migrate")
    
    for filepath in miriade_files:
        try:
            safe_designation = extract_safe_designation(filepath)
            
            # Check if already in database
            if db_utils.miriade_data_exists(safe_designation):
                logger.info(f"Miriade data for {safe_designation} already exists in database, skipping")
                continue
            
            # Read CSV file
            logger.info(f"Migrating Miriade data: {filepath}")
            df = pd.read_csv(filepath)
            
            # Save to database
            db_utils.save_miriade_data(safe_designation, df)
            logger.info(f"Successfully migrated {len(df)} Miriade records for {safe_designation}")
            
        except Exception as e:
            logger.error(f"Error migrating {filepath}: {str(e)}")


def migrate_ztf_data():
    """Migrate ZTF CSV files to DuckDB."""
    ztf_files = glob.glob('./db/ztf/*.csv.gz') + glob.glob('./db/ztf/*.csv')
    
    logger.info(f"Found {len(ztf_files)} ZTF files to migrate")
    
    for filepath in ztf_files:
        try:
            safe_designation = extract_safe_designation(filepath)
            
            # Check if already in database
            if db_utils.ztf_data_exists(safe_designation):
                logger.info(f"ZTF data for {safe_designation} already exists in database, skipping")
                continue
            
            # Read CSV file
            logger.info(f"Migrating ZTF data: {filepath}")
            df = pd.read_csv(filepath)
            
            # Save to database
            db_utils.save_ztf_data(safe_designation, df)
            logger.info(f"Successfully migrated {len(df)} ZTF records for {safe_designation}")
            
        except Exception as e:
            logger.error(f"Error migrating {filepath}: {str(e)}")


def main():
    """Run the migration."""
    logger.info("Starting CSV to DuckDB migration")
    
    # Initialize databases
    logger.info("Initializing DuckDB databases")
    db_utils.init_all_databases()
    
    # Migrate each data source
    migrate_mpc_data()
    migrate_miriade_data()
    migrate_ztf_data()
    
    logger.info("Migration complete!")
    
    # Print summary
    asteroids = db_utils.list_all_asteroids()
    logger.info(f"Total asteroids in metadata database: {len(asteroids)}")
    
    # Note about metadata
    logger.info("\nNote: Asteroid metadata (designations, names) will be populated")
    logger.info("automatically when asteroids are queried through the application.")


if __name__ == '__main__':
    main()
