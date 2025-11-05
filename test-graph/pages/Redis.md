tags:: #database, #cache
type:: database
created:: [[2024-01-26]]

- Redis is an in-memory data structure store
  - Key-value database
  - Used for caching, sessions, queues
  - Extremely fast (sub-millisecond latency)
- Data structures
  id:: 65a2c1f0-7777-4567-89ab-cdef01234571
  - Strings (simple values)
  - Lists (ordered collections)
  - Sets (unique values)
  - Sorted sets (scored members)
  - Hashes (field-value pairs)
  - Bitmaps and HyperLogLogs
  - Streams (log data structure)
- Common use cases
  - Caching frequently accessed data
    - Reduces database load
  - Session storage
  - Real-time analytics
  - Pub/sub messaging
  - Rate limiting
  - Leaderboards with sorted sets
- Persistence options
  - RDB: point-in-time snapshots
  - AOF: append-only file
  - Can combine both
  - Or use as pure cache (no persistence)
- Performance tips
  - Use pipelining for bulk operations
  - Avoid KEYS command in production
  - Set expiration times on keys
  - Use Redis Cluster for horizontal scaling
- Integration with [[Python]]
  - redis-py client library
  - Used by [[Django]] for caching
  - Celery uses Redis as message broker
- Limitations
  - Data must fit in memory
  - Single-threaded (one command at a time)
  - Not ideal for complex queries
