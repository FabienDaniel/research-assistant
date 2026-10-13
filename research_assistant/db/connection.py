import time

import oracledb


def connect_to_oracle(max_retries=3, retry_delay=5, user="system", password="YourPassword123",
                      dsn="127.0.0.1:1521/FREE", program="langchain_oracledb_deep_research_demo"):
    """
    Connect to Oracle database with retry logic and better error handling.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retries
    """

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Connection attempt {attempt}/{max_retries}...")
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                program=program
            )
            print("✓ Connected successfully!")

            # Test the connection
            with conn.cursor() as cur:
                try:
                    cur.execute("SELECT banner FROM v$version WHERE banner LIKE 'Oracle%';")
                    banner = cur.fetchone()[0]
                    print(f"\n{banner}")
                except oracledb.Error:
                    cur.execute("SELECT 1 FROM DUAL")
                    cur.fetchone()
                    print("  Connected to Oracle Database")

            return conn

        except oracledb.OperationalError as e:
            error_msg = str(e)
            print(f"✗ Connection failed (attempt {attempt}/{max_retries})")

            if "DPY-4011" in error_msg or "Connection reset by peer" in error_msg:
                print("  → This usually means:")
                print("    1. Database is still starting up (wait 2-3 minutes)")
                print("    2. Listener configuration issue")
                print("    3. Container is not running")

                if attempt < max_retries:
                    print(f"\n  Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    print("\n  💡 Try running: setup_oracle_database()")
                    print("     This will fix the listener and verify the connection.")
                    raise
            else:
                raise
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            raise

    raise ConnectionError("Failed to connect after all retries")
