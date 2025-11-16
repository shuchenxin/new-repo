"""
Configuration settings for the Flask web application.
"""

import os


class Config:
    """Base configuration class."""

    # MongoDB configuration
    # Default assumes docker-compose service name "mongodb"
    # For local development, set MONGO_URI=mongodb://localhost:27017/ in .env
    # For docker-compose, this default works if service is named "mongodb"
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017/")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ml_system")

    # Flask configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    API_LIMIT = int(os.getenv("API_LIMIT", "100"))
