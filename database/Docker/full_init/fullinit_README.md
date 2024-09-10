# Full Database Initialization

This directory contains the necessary files to perform a full initialization of the Vocabulo project database from scratch.

## Contents

- `docker-compose.yml`: Docker Compose file for setting up the database and pgAdmin
- `.env.example`: Example environment file (copy to `.env` and fill in your values)
- SQL scripts in the `sql/` directory for creating and populating the database

## Automated Build Process

1. **Environment Setup**:
   - Copy `.env.example` to `.env`
   - Fill in the necessary environment variables in `.env`

2. **Docker Compose**:
   - Builds and starts PostgreSQL and pgAdmin containers
   - Mounts SQL scripts and data directories

3. **Database Initialization**:
   - SQL scripts are executed in alphabetical order
   - Creates database schema, tables, and relationships
   - Imports initial data (if available)

4. **Post-Initialization**:
   - Sets up user roles and permissions

## Usage

1. Ensure Docker and Docker Compose are installed on your system
2. Navigate to this directory
3. Run:
   ```
   docker-compose up -d
   ```
4. The database will be initialized and available at localhost:5432
5. pgAdmin will be available at localhost:5050

## Notes

- This process creates a clean, fully initialized database
- Suitable for development and testing environments
- Requires SQL scripts and initial data files (not included in repository)

For more detailed information about the database structure and data sources, please refer to the main project README.
