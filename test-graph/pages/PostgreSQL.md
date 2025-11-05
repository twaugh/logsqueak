tags:: #database, #sql
type:: database
created:: [[2024-01-25]]

- PostgreSQL is an open-source relational database
  - ACID compliant
  - Highly extensible
  - Strong standards compliance
- Key features
  id:: 65a2c1f0-6666-4567-89ab-cdef01234570
  - Advanced data types
    - JSON and JSONB
    - Arrays
    - Custom types
    - UUID support
  - Full-text search
  - Foreign data wrappers
  - Table partitioning
  - Row-level security
- Performance features
  - Query planner and optimizer
  - Indexes: B-tree, Hash, GiST, GIN
  - Materialized views
  - Parallel query execution
  - Connection pooling with pgBouncer
- JSONB advantages
  - Binary storage format
  - Faster processing than JSON
  - Indexable with GIN indexes
  - Query with JSON operators
- Common patterns
  - Use SERIAL or IDENTITY for auto-increment
  - Create indexes on foreign keys
  - Use EXPLAIN ANALYZE for query tuning
  - Regular VACUUM for maintenance
- Replication
  - Streaming replication
  - Logical replication
  - Hot standby servers
- Use with [[Python]]
  - psycopg2 driver
  - SQLAlchemy ORM
  - [[Django]] ORM support
