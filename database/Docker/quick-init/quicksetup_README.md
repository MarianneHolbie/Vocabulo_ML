# Quick Database Initialization

This directory contains files for a quick initialization of the Vocabulo project database using a pre-made dump.

## Contents

- `docker-compose.yml`: Docker Compose file for setting up the database and pgAdmin
- `.env.example`: Example environment file (copy to `.env` and fill in your values)

## Quick Start Process

1. Copy `.env.example` to `.env` and fill in the values
2. Ensure you have the `dump.sql` file in this directory
3. Run:
   ```
   docker-compose up -d
   ```
4. The database will be initialized with the pre-made dump

## Notes

- This method is faster but less flexible than the full initialization
- Suitable for quick setups or demonstrations
- Requires the database dump file (not included in the repository)

For more detailed information about the project, database structure, and full setup process, please refer to 
the main project repository at [Vocabulo Project Repository URL](https://github.com/TessierV/vocabulo).