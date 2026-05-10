-- Initial PostgreSQL setup for ECOT AI Backend
-- This file is run automatically by the postgres Docker image on first start.
-- SQLAlchemy will create the actual tables via create_all(), but we set up
-- useful extensions and the database locale here.

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- for uuid_generate_v4()
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";  -- for query monitoring

-- Performance settings (applied as session defaults; adjust in postgresql.conf for prod)
ALTER DATABASE ecot_db SET timezone TO 'UTC';
