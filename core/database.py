"""
Database management for AI Service
"""

import asyncio
import asyncpg
import logging
from typing import Dict, List, Any, Optional


class DatabaseManager:
    """Manages database connections and queries"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
    
    async def connect(self):
        """Initialize database connection pool"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            self.logger.info("✅ Database connection pool created")
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            self.logger.info("Database connection pool closed")
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.connection_pool is not None
    
    async def fetch_stations(self) -> List[Dict[str, Any]]:
        """Fetch all stations from database"""
        if not self.connection_pool:
            raise Exception("Database not connected")
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, name, station_address, latitude, longitude, 
                       gtfs_source, is_bus_stop, created_at
                FROM stations_station
                ORDER BY name
            """)
            
            return [dict(row) for row in rows]
    
    async def fetch_routes(self) -> List[Dict[str, Any]]:
        """Fetch all routes from database"""
        if not self.connection_pool:
            raise Exception("Database not connected")
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, short_name, long_name, source, route_type, created_at
                FROM stations_route
                ORDER BY short_name
            """)
            
            return [dict(row) for row in rows]
    
    async def fetch_trips(self) -> List[Dict[str, Any]]:
        """Fetch all trips from database"""
        if not self.connection_pool:
            raise Exception("Database not connected")
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, route_id, trip_headsign, direction_id, created_at
                FROM stations_trip
                ORDER BY id
            """)
            
            return [dict(row) for row in rows]
    
    async def fetch_fares(self) -> List[Dict[str, Any]]:
        """Fetch all fares from database"""
        if not self.connection_pool:
            raise Exception("Database not connected")
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, fare_amount, transport_type, created_at
                FROM stations_fare
                ORDER BY fare_amount
            """)
            
            return [dict(row) for row in rows]
    
    async def search_stations(self, query: str) -> List[Dict[str, Any]]:
        """Search stations by name or address"""
        if not self.connection_pool:
            raise Exception("Database not connected")
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, name, station_address, latitude, longitude, gtfs_source
                FROM stations_station
                WHERE name ILIKE $1 OR station_address ILIKE $1
                ORDER BY name
                LIMIT 20
            """, f"%{query}%")
            
            return [dict(row) for row in rows]
