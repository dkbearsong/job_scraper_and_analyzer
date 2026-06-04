import psycopg2
from psycopg2 import sql
from psycopg2.extras import DictCursor
from typing import Dict, Iterable, Optional, Any, List, Tuple


class PostgresManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "",
        dbname: Optional[str] = None,
        connect_now: bool = False,
        default_db: str = "postgres",
    ):
        self.conn_params = dict(host=host, port=port, user=user, password=password)
        self.dbname = dbname
        self.default_db = default_db
        self._conn = None
        if connect_now and dbname:
            self.connect(dbname)

    def connect(self, dbname: Optional[str] = None):
        """Open a connection to a specific database (or to self.dbname if not provided)."""
        target = dbname or self.dbname
        if target is None:
            raise ValueError("No database name provided to connect to.")
        if self._conn:
            self._conn.close()
        params = dict(self.conn_params, dbname=target)
        params["port"] = str(params["port"])
        self._conn = psycopg2.connect(**params) # type: ignore
        self._conn.autocommit = False
        return self._conn

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _get_conn(self, dbname: Optional[str] = None):
        """Return an open connection (connects automatically if not connected)."""
        target = dbname or self.dbname or self.default_db
        if self._conn is None or self._conn.closed:
            self.connect(target)
        if self._conn is None:
            raise RuntimeError("Failed to establish database connection.")
        return self._conn
    
    def database_exists(self, dbname: str) -> bool:
        """
        Check whether a database exists.
        Must connect to default_db (usually 'postgres').
        """
        temp_conn = psycopg2.connect(**dict(self.conn_params, dbname=self.default_db))  # type: ignore
        temp_conn.autocommit = True
        try:
            with temp_conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (dbname,),
                )
                return cur.fetchone() is not None
        finally:
            temp_conn.close()

    def create_database(self, new_dbname: str):
        """
        Create a database if it doesn't already exist.
        Connects first to default_db with provided credentials.
        """
        temp_conn = psycopg2.connect(**dict(self.conn_params, dbname=self.default_db))  # type: ignore
        temp_conn.autocommit = True
        try:
            with temp_conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (new_dbname,),
                )
                exists = cur.fetchone() is not None
                if not exists:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(new_dbname)))
        finally:
            temp_conn.close()

    def execute_sql(self, statement: str, params: Optional[Iterable[Any]] = None, dbname: Optional[str] = None, fetch: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Execute arbitrary SQL. Returns rows as list of dicts if fetch=True. For statement, use %s placeholders for params.
        NOTE: Use for ad-hoc SQL; prefer specialized methods for inserts/updates/selects.

        """
        conn = self._get_conn(dbname)
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(statement, params)
            if fetch:
                rows = cur.fetchall()
                return rows
            conn.commit()
        return None

    def create_table(
            self,
            table_name: str,
            columns: Dict[str, str],
            if_not_exists: bool = True,
            dbname: Optional[str] = None
        ):
        """
        columns: dict mapping column_name -> column definition string
        e.g. {"id": "SERIAL PRIMARY KEY", "name": "TEXT NOT NULL"}
        """
        parts = []
        for colname, definition in columns.items():
            # definition is raw SQL fragment (type, constraints). It's expected to be provided by developer.
            parts.append(sql.SQL("{} {}").format(sql.Identifier(colname), sql.SQL(definition)))
        query = sql.SQL("CREATE TABLE {ine} {tbl} ({cols})").format(
            ine=sql.SQL("IF NOT EXISTS") if if_not_exists else sql.SQL(""),
            tbl=sql.Identifier(table_name),
            cols=sql.SQL(", ").join(parts),
        )
        conn = self._get_conn(dbname)
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()

    def add_foreign_key(
        self,
        src_table: str,
        src_cols: Iterable[str],
        ref_table: str,
        ref_cols: Iterable[str],
        constraint_name: Optional[str] = None,
        on_delete: Optional[str] = None,
        dbname: Optional[str] = None,
    ):
        """
        Add foreign key constraint:
        src_cols and ref_cols are iterables (usually single-column).
        constraint_name: optional name for the constraint
        on_delete: optional action (e.g. "CASCADE", "SET NULL")
        """
        src_cols = list(src_cols)
        ref_cols = list(ref_cols)
        if len(src_cols) != len(ref_cols):
            raise ValueError("src_cols and ref_cols must have same length.")
        if not constraint_name:
            constraint_name = f"fk_{src_table}_{'_'.join(src_cols)}__{ref_table}_{'_'.join(ref_cols)}"

        query = sql.SQL("ALTER TABLE {src} ADD CONSTRAINT {cname} FOREIGN KEY ({src_cols}) REFERENCES {ref} ({ref_cols}) {on_delete}").format(
            src=sql.Identifier(src_table),
            cname=sql.Identifier(constraint_name),
            src_cols=sql.SQL(", ").join([sql.Identifier(c) for c in src_cols]),
            ref=sql.Identifier(ref_table),
            ref_cols=sql.SQL(", ").join([sql.Identifier(c) for c in ref_cols]),
            on_delete=sql.SQL(f"ON DELETE {on_delete}") if on_delete else sql.SQL(""),
        )
        conn = self._get_conn(dbname)
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()

    def insert(self, table: str, data: Dict[str, Any], returning: Optional[Iterable[str]] = None, dbname: Optional[str] = None) -> Optional[List[Tuple]]:
        """
        Insert a row (or many rows if data is list-of-dicts can be handled by repeating calls or extended).
        Returns returning rows if requested.
        """
        keys = list(data.keys())
        values = [data[k] for k in keys]
        placeholders = sql.SQL(", ").join(sql.Placeholder() * len(keys))
        query = sql.SQL("INSERT INTO {tbl} ({fields}) VALUES ({values})").format(
            tbl=sql.Identifier(table),
            fields=sql.SQL(", ").join(map(sql.Identifier, keys)),
            values=placeholders,
        )
        if returning:
            ret_fields = sql.SQL(", ").join(map(sql.Identifier, returning))
            query = sql.SQL("{} RETURNING {}").format(query, ret_fields)

        conn = self._get_conn(dbname)
        with conn.cursor() as cur:
            cur.execute(query, values)
            result = None
            if returning:
                result = cur.fetchall()
            conn.commit()
            return result

    def update(self, table: str, set_values: Dict[str, Any], where: Dict[str, Any], returning: Optional[Iterable[str]] = None, dbname: Optional[str] = None) -> Optional[List[Tuple]]:
        """
        Build and run an UPDATE ... SET ... WHERE ...
        where: mapping for equality conditions joined by AND
        """
        set_keys = list(set_values.keys())
        where_keys = list(where.keys())

        set_frag = sql.SQL(", ").join(
            sql.SQL("{} = {}").format(sql.Identifier(k), sql.Placeholder()) for k in set_keys
        )

        where_frags = []
        where_params = []
        for k in where_keys:
            value = where[k]
            if isinstance(value, (list, tuple, set)):
                if not value:
                    raise ValueError(f"WHERE parameter '{k}' list cannot be empty")
                placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in value)
                where_frags.append(
                    sql.SQL("{} IN ({})").format(sql.Identifier(k), placeholders)
                )
                where_params.extend(value)
            else:
                where_frags.append(
                    sql.SQL("{} = {}").format(sql.Identifier(k), sql.Placeholder())
                )
                where_params.append(value)

        where_frag = sql.SQL(" AND ").join(where_frags)

        query = sql.SQL("UPDATE {tbl} SET {setc} WHERE {wher}").format(
            tbl=sql.Identifier(table),
            setc=set_frag,
            wher=where_frag,
        )
        params = list(set_values[k] for k in set_keys) + where_params

        if returning:
            ret_fields = sql.SQL(", ").join(map(sql.Identifier, returning))
            query = sql.SQL("{} RETURNING {}").format(query, ret_fields)

        conn = self._get_conn(dbname)
        with conn.cursor() as cur:
            cur.execute(query, params)
            result = None
            if returning:
                result = cur.fetchall()
            conn.commit()
            return result

    def search(self, table: str, where: Optional[Dict[str, Any]] = None, columns: Optional[Iterable[str]] = None, limit: Optional[int] = None, dbname: Optional[str] = None) -> List[Tuple]:
        """
        Simple SELECT helper. where is dict of equality conditions joined by AND.
        columns: list or None (means '*')
        """
        cols = sql.SQL(", ").join(map(sql.Identifier, columns)) if columns else sql.SQL("*")
        base = sql.SQL("SELECT {cols} FROM {tbl}").format(cols=cols, tbl=sql.Identifier(table))
        params = []
        if where:
            where_keys = list(where.keys())
            where_frag = sql.SQL(" AND ").join(
                sql.SQL("{} = {}").format(sql.Identifier(k), sql.Placeholder()) for k in where_keys
            )
            base = sql.SQL("{} WHERE {}").format(base, where_frag)
            params = [where[k] for k in where_keys]
        if limit:
            base = sql.SQL("{} LIMIT {}").format(base, sql.Literal(limit))

        conn = self._get_conn(dbname)
        # print(f"base: {base} | params: {params}")
        with conn.cursor() as cur:
            cur.execute(base, params)
            rows = cur.fetchall()
            return rows


# Example usage when run as script
if __name__ == "__main__":
    mgr = PostgresManager(host="localhost", port=5432, user="postgres", password="postgres", dbname="demo_db")
    # 1) create database (connects to default "postgres" to create)
    mgr.create_database("demo_db")

    # 2) connect to database
    mgr.connect()

    # 3) create tables
    mgr.create_table("users", {
        "id": "SERIAL PRIMARY KEY",
        "username": "TEXT NOT NULL UNIQUE",
        "email": "TEXT"
    })

    mgr.create_table("posts", {
        "id": "SERIAL PRIMARY KEY",
        "user_id": "INTEGER NOT NULL",
        "title": "TEXT",
        "body": "TEXT"
    })

    # 4) add foreign key
    mgr.add_foreign_key("posts", ["user_id"], "users", ["id"], on_delete="CASCADE")

    # 5) insert user and post
    user_ret = mgr.insert("users", {"username": "alice", "email": "alice@example.com"}, returning=["id"])
    user_id = user_ret[0][0] if user_ret else None

    post_ret = mgr.insert("posts", {"user_id": user_id, "title": "Hello", "body": "First post"}, returning=["id"])
    print("Inserted user id:", user_id, "post id:", post_ret)

    # 6) search
    users = mgr.search("users")
    print("Users:", users)

    # 7) update
    mgr.update("users", {"email": "alice@newdomain.com"}, {"id": user_id})

    mgr.close()