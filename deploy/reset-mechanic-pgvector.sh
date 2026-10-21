#!/usr/bin/env bash
# Reset the "mechanic" Postgres role/database for the OGX remote::pgvector
# provider, and install the pgvector extension in it.
#
# Run this AS the postgres superuser, on the db server itself (uses the
# local socket / peer auth via `psql -U postgres`, no -h/-p needed). If you
# need to run it remotely instead, add `-h db.home.glroland.com -p 5432` to
# each psql invocation below (and make sure pg_hba.conf allows postgres to
# connect that way).
#
# DB_NAME/DB_USER/DB_PASSWORD can be overridden via the environment; the
# defaults below must match the mechanic-pgvector k8s Secret:
#   oc create secret generic mechanic-pgvector -n mechanic \
#     --from-literal=database-name=mechanic \
#     --from-literal=database-user=mechanic \
#     --from-literal=database-password=mechanic123

set -euo pipefail

DB_NAME="${DB_NAME:-mechanic}"
DB_USER="${DB_USER:-mechanic}"
DB_PASSWORD="${DB_PASSWORD:-mechanic123}"

echo "==> Terminating any existing connections to '${DB_NAME}'..."
psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = '${DB_NAME}' AND pid <> pg_backend_pid();
"

echo "==> Dropping database '${DB_NAME}' (if it exists)..."
psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS ${DB_NAME};"

echo "==> Dropping role '${DB_USER}' (if it exists)..."
psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c "DROP ROLE IF EXISTS ${DB_USER};"

echo "==> Creating role '${DB_USER}'..."
psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';"

echo "==> Creating database '${DB_NAME}' owned by '${DB_USER}'..."
psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE ${DB_NAME} WITH OWNER ${DB_USER};"

echo "==> Granting '${DB_USER}' superuser (needed to CREATE EXTENSION; Postgres has no separate built-in 'dba' role)..."
psql -U postgres -d postgres -v ON_ERROR_STOP=1 -c "ALTER ROLE ${DB_USER} WITH SUPERUSER CREATEDB CREATEROLE;"

echo "==> Creating the vector extension in '${DB_NAME}'..."
psql -U postgres -d "${DB_NAME}" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "==> Verifying..."
psql -U postgres -d "${DB_NAME}" -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"

echo "==> Done."
