import oracledb

CONVERSATIONAL_TABLE = "CONVERSATIONAL_MEMORY"
KNOWLEDGE_BASE_TABLE = "SEMANTIC_MEMORY"
WORKFLOW_TABLE = "WORKFLOW_MEMORY"
TOOLBOX_TABLE = "TOOLBOX_MEMORY"
ENTITY_TABLE = "ENTITY_MEMORY"
SUMMARY_TABLE = "SUMMARY_MEMORY"
TOOL_LOG_TABLE = "TOOL_LOG_MEMORY"

ALL_TABLES = [
    CONVERSATIONAL_TABLE,
    KNOWLEDGE_BASE_TABLE,
    WORKFLOW_TABLE,
    TOOLBOX_TABLE,
    ENTITY_TABLE,
    SUMMARY_TABLE,
    TOOL_LOG_TABLE,
]


def drop_tables(conn, tables):
    """Drop the given tables, discarding their data. A missing table is not an error."""
    for table in tables:
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE {table} PURGE")
                print(f"  - {table} (dropped)")
        except oracledb.Error as e:
            if "ORA-00942" in str(e):
                print(f"  - {table} (not exists)")
            else:
                print(f"  ✗ {table}: {e}")

    conn.commit()


def table_exists(conn, table_name):
    """Check if a table exists in the current user's schema."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) 
            FROM USER_TABLES 
            WHERE TABLE_NAME = UPPER(:table_name)
        """, {"table_name": table_name})
        return cur.fetchone()[0] > 0


def create_conversational_history_table(conn, table_name: str = "CONVERSATIONAL_MEMORY"):
    """
    Create a table to store conversational history.
    If the table already exists, returns the table name without recreating it.
    """
    # Check if table already exists
    if table_exists(conn, table_name):
        print(f"  ⏭️ Table {table_name} already exists (using existing table)")
        return table_name

    with conn.cursor() as cur:
        # Create table with proper schema
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id VARCHAR2(100) DEFAULT SYS_GUID() PRIMARY KEY,
                thread_id VARCHAR2(100) NOT NULL,
                role VARCHAR2(50) NOT NULL,
                content CLOB NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata CLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary_id VARCHAR2(100) DEFAULT NULL
            )
        """)

        # Create index on thread_id for faster lookups
        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_thread_id ON {table_name}(thread_id)
        """)

        # Create index on timestamp for ordering
        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_timestamp ON {table_name}(timestamp)
        """)

    conn.commit()
    print(f"  ✅ Table {table_name} created successfully with indexes")
    return table_name


def create_tool_log_table(conn, table_name: str = "TOOL_LOG_MEMORY"):
    """
    Create a table to store raw tool execution logs per thread.
    If the table already exists, returns the table name without recreating it.
    """
    if table_exists(conn, table_name):
        print(f"  ⏭️ Table {table_name} already exists (using existing table)")
        return table_name

    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id VARCHAR2(100) DEFAULT SYS_GUID() PRIMARY KEY,
                thread_id VARCHAR2(100) NOT NULL,
                tool_call_id VARCHAR2(200),
                tool_name VARCHAR2(200) NOT NULL,
                tool_args CLOB,
                result CLOB,
                result_preview VARCHAR2(2000),
                status VARCHAR2(30) DEFAULT 'success',
                error_message CLOB,
                metadata CLOB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_thread_id ON {table_name}(thread_id)
        """)
        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_tool_name ON {table_name}(tool_name)
        """)
        cur.execute(f"""
            CREATE INDEX idx_{table_name.lower()}_timestamp ON {table_name}(timestamp)
        """)

    conn.commit()
    print(f"  ✅ Table {table_name} created successfully with indexes")
    return table_name


def safe_create_index(conn, vs, idx_name):
    """Create IVF vector index using raw SQL for maximum compatibility.

    Uses IVF (NEIGHBOR PARTITIONS) instead of HNSW to avoid:
    - ORA-00600 on some Oracle Free versions
    - ORA-51928 (DML not supported with INMEMORY NEIGHBOR GRAPH)
    - ORA-51962 (vector memory pool sizing issues)

    Handles ORA-00955 (index already exists) by skipping.
    """
    dist_map = {
        "COSINE": "COSINE",
        "EUCLIDEAN_DISTANCE": "EUCLIDEAN",
        "DOT_PRODUCT": "DOT",
    }
    dist = dist_map.get(vs.distance_strategy.name, "COSINE")

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE VECTOR INDEX {idx_name}"
                f" ON {vs.table_name}(EMBEDDING)"
                f" ORGANIZATION NEIGHBOR PARTITIONS"
                f" DISTANCE {dist}"
                f" WITH TARGET ACCURACY 95"
            )
        print(f"  ✅ Created index: {idx_name}")
    except Exception as e:
        err = str(e)
        if "ORA-00955" in err:
            print(f"  ⏭️  Index already exists: {idx_name}")
        else:
            raise
