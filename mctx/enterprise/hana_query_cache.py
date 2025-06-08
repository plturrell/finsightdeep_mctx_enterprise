# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Query caching and optimization for SAP HANA integration."""

import copy
import datetime
import functools
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np

from mctx.enterprise.hana_integration import HanaConnection

# Type variables
T = TypeVar('T')
QueryResult = TypeVar('QueryResult')

# Constants - configurable via environment variables
DEFAULT_CACHE_TTL = int(os.environ.get('MCTX_QUERY_CACHE_TTL', '300'))  # 5 minutes
DEFAULT_CACHE_SIZE = int(os.environ.get('MCTX_QUERY_CACHE_SIZE', '1000'))  # Maximum cache size
DEFAULT_QUERY_TIMEOUT = int(os.environ.get('MCTX_QUERY_TIMEOUT', '60'))  # 1 minute
CACHE_ENABLED = os.environ.get('MCTX_QUERY_CACHE_ENABLED', 'true').lower() == 'true'
CACHE_CLEANUP_INTERVAL = int(os.environ.get('MCTX_CACHE_CLEANUP_INTERVAL', '60'))  # seconds

# Configure logging
logger = logging.getLogger("mctx.enterprise.hana_cache")


class QueryCacheEntry:
    """A single entry in the query cache."""
    
    def __init__(self, query: str, params: Tuple, result: Any, ttl: int = DEFAULT_CACHE_TTL):
        """Initialize a new cache entry.
        
        Args:
            query: The SQL query string.
            params: The query parameters.
            result: The query result to cache.
            ttl: Time-to-live in seconds.
        """
        self.query = query
        self.params = params
        self.result = result
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 1
        self.last_accessed = self.created_at
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired.
        
        Returns:
            True if the entry has expired, False otherwise.
        """
        return (time.time() - self.created_at) > self.ttl
    
    def access(self) -> None:
        """Record an access to this cache entry."""
        self.access_count += 1
        self.last_accessed = time.time()
        

class QueryCacheKey:
    """A key for the query cache, based on the query and parameters."""
    
    def __init__(self, query: str, params: Tuple):
        """Initialize a new cache key.
        
        Args:
            query: The SQL query string.
            params: The query parameters.
        """
        self.query = query
        self.params = params
        
        # Precompute hash for faster lookup
        self._hash = self._compute_hash()
    
    def _compute_hash(self) -> int:
        """Compute a hash for this cache key.
        
        Returns:
            A hash of the query and parameters.
        """
        # Normalize the query by removing extra whitespace
        normalized_query = ' '.join(self.query.split())
        
        # Create a hash of the query and parameters
        hasher = hashlib.md5()
        hasher.update(normalized_query.encode('utf-8'))
        
        # Add parameters to the hash
        for param in self.params:
            # Handle different parameter types
            if isinstance(param, (str, int, float, bool, type(None))):
                hasher.update(str(param).encode('utf-8'))
            elif isinstance(param, (datetime.date, datetime.datetime)):
                hasher.update(str(param.isoformat()).encode('utf-8'))
            elif isinstance(param, (list, tuple)):
                hasher.update(str(param).encode('utf-8'))
            else:
                # For complex objects, use their string representation
                hasher.update(str(param).encode('utf-8'))
        
        # Return the hash as an integer
        return int(hasher.hexdigest(), 16)
    
    def __hash__(self) -> int:
        """Get the hash of this cache key.
        
        Returns:
            The precomputed hash.
        """
        return self._hash
    
    def __eq__(self, other: Any) -> bool:
        """Check if this cache key is equal to another.
        
        Args:
            other: The other cache key to compare with.
            
        Returns:
            True if the keys are equal, False otherwise.
        """
        if not isinstance(other, QueryCacheKey):
            return False
        
        return (self.query == other.query and self.params == other.params)


class QueryCache:
    """A cache for database queries to reduce load on SAP HANA."""
    
    def __init__(self, 
                max_size: int = DEFAULT_CACHE_SIZE, 
                default_ttl: int = DEFAULT_CACHE_TTL,
                enabled: bool = CACHE_ENABLED):
        """Initialize a new query cache.
        
        Args:
            max_size: Maximum number of items to cache.
            default_ttl: Default time-to-live in seconds.
            enabled: Whether the cache is enabled.
        """
        self._cache: Dict[QueryCacheKey, QueryCacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._enabled = enabled
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._invalidations = 0
        self._inserts = 0
        
        # For tracking frequently accessed tables
        self._table_access_count: Dict[str, int] = {}
        
        # Start a background thread to clean expired entries
        if self._enabled:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self) -> None:
        """Start a background thread to clean expired entries."""
        def cleanup_thread():
            while True:
                time.sleep(CACHE_CLEANUP_INTERVAL)
                try:
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in cache cleanup thread: {e}")
        
        thread = threading.Thread(target=cleanup_thread, daemon=True)
        thread.start()
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        with self._lock:
            # Find expired keys
            expired_keys = [
                key for key, entry in self._cache.items() 
                if entry.is_expired()
            ]
            
            # Remove expired entries
            for key in expired_keys:
                del self._cache[key]
                self._evictions += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _evict_if_needed(self) -> None:
        """Evict entries if the cache is full."""
        with self._lock:
            if len(self._cache) >= self._max_size:
                # First try to evict expired entries
                expired_keys = [
                    key for key, entry in self._cache.items() 
                    if entry.is_expired()
                ]
                
                for key in expired_keys:
                    del self._cache[key]
                    self._evictions += 1
                
                # If still over max size, evict least recently used entries
                if len(self._cache) >= self._max_size:
                    entries = sorted(
                        self._cache.items(), 
                        key=lambda item: (item[1].last_accessed, -item[1].access_count)
                    )
                    
                    # Remove 10% of entries or at least one
                    num_to_remove = max(1, len(entries) // 10)
                    for i in range(num_to_remove):
                        key, _ = entries[i]
                        del self._cache[key]
                        self._evictions += 1
                    
                    logger.debug(f"Evicted {num_to_remove} least recently used cache entries")
    
    def get(self, query: str, params: Tuple) -> Optional[Any]:
        """Get a result from the cache.
        
        Args:
            query: The SQL query string.
            params: The query parameters.
            
        Returns:
            The cached result, or None if not found or expired.
        """
        if not self._enabled:
            return None
        
        key = QueryCacheKey(query, params)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None or entry.is_expired():
                self._misses += 1
                
                # Remove expired entry if it exists
                if entry is not None:
                    del self._cache[key]
                
                return None
            
            # Update access statistics
            entry.access()
            self._hits += 1
            
            # Track table access - very simple parsing for demonstration
            for table in self._extract_tables_from_query(query):
                self._table_access_count[table] = self._table_access_count.get(table, 0) + 1
            
            # Return a copy to avoid modifications affecting the cache
            return copy.deepcopy(entry.result)
    
    def set(self, query: str, params: Tuple, result: Any, ttl: Optional[int] = None) -> None:
        """Store a result in the cache.
        
        Args:
            query: The SQL query string.
            params: The query parameters.
            result: The result to cache.
            ttl: Time-to-live in seconds (defaults to class default).
        """
        if not self._enabled:
            return
        
        # Check if we need to evict entries
        self._evict_if_needed()
        
        key = QueryCacheKey(query, params)
        ttl = ttl if ttl is not None else self._default_ttl
        
        with self._lock:
            # Store a copy to avoid modifications affecting the cache
            self._cache[key] = QueryCacheEntry(
                query, params, copy.deepcopy(result), ttl
            )
            self._inserts += 1
    
    def invalidate(self, query_pattern: Optional[str] = None, table_names: Optional[List[str]] = None) -> int:
        """Invalidate entries matching a query pattern or table names.
        
        Args:
            query_pattern: A pattern to match against queries.
            table_names: List of table names to match.
            
        Returns:
            The number of entries invalidated.
        """
        if not self._enabled:
            return 0
        
        with self._lock:
            if query_pattern is None and table_names is None:
                # Invalidate all entries
                count = len(self._cache)
                self._cache.clear()
                self._invalidations += count
                return count
            
            # Find keys matching the pattern or table names
            matching_keys = []
            
            for key in self._cache.keys():
                should_invalidate = False
                
                if query_pattern and query_pattern in key.query:
                    should_invalidate = True
                
                if table_names:
                    for table in table_names:
                        if self._is_query_using_table(key.query, table):
                            should_invalidate = True
                            break
                
                if should_invalidate:
                    matching_keys.append(key)
            
            # Remove matching entries
            for key in matching_keys:
                del self._cache[key]
            
            self._invalidations += len(matching_keys)
            return len(matching_keys)
    
    def clear(self) -> int:
        """Clear all entries from the cache.
        
        Returns:
            The number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._table_access_count.clear()
            self._invalidations += count
            return count
    
    def _extract_tables_from_query(self, query: str) -> Set[str]:
        """Extract table names from a query.
        
        This is a simple heuristic implementation. For production use,
        consider using a proper SQL parser.
        
        Args:
            query: The SQL query string.
            
        Returns:
            A set of table names.
        """
        tables = set()
        query_upper = query.upper()
        
        # Look for common SQL patterns
        from_index = query_upper.find("FROM ")
        if from_index != -1:
            # Extract the table after FROM
            rest = query_upper[from_index + 5:]
            # Find the first non-table part
            for delimiter in [" WHERE ", " JOIN ", " GROUP ", " ORDER ", " HAVING ", " LIMIT ", " OFFSET "]:
                end_index = rest.find(delimiter)
                if end_index != -1:
                    rest = rest[:end_index]
            
            # Clean up and add the table
            table = rest.strip().split()[0]
            tables.add(table)
        
        # Look for JOIN clauses
        join_index = 0
        while True:
            join_index = query_upper.find("JOIN ", join_index)
            if join_index == -1:
                break
            
            # Extract the table after JOIN
            rest = query_upper[join_index + 5:]
            # Find the first non-table part
            for delimiter in [" ON ", " WHERE ", " JOIN ", " GROUP ", " ORDER ", " HAVING ", " LIMIT ", " OFFSET "]:
                end_index = rest.find(delimiter)
                if end_index != -1:
                    rest = rest[:end_index]
            
            # Clean up and add the table
            table = rest.strip().split()[0]
            tables.add(table)
            
            # Move to the next position
            join_index += 5
        
        return tables
    
    def _is_query_using_table(self, query: str, table_name: str) -> bool:
        """Check if a query uses a specific table.
        
        Args:
            query: The SQL query string.
            table_name: The table name to check.
            
        Returns:
            True if the query uses the table, False otherwise.
        """
        query_upper = query.upper()
        table_upper = table_name.upper()
        
        # Common patterns for table references
        patterns = [
            f"FROM {table_upper} ",
            f"FROM {table_upper},",
            f"FROM {table_upper}.",
            f"FROM {table_upper}\n",
            f"FROM {table_upper})",
            f"JOIN {table_upper} ",
            f"JOIN {table_upper},",
            f"JOIN {table_upper}.",
            f"JOIN {table_upper}\n",
            f"JOIN {table_upper})",
            f"UPDATE {table_upper} ",
            f"INSERT INTO {table_upper} ",
            f"DELETE FROM {table_upper} ",
            f"MERGE INTO {table_upper} "
        ]
        
        for pattern in patterns:
            if pattern in query_upper:
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            A dictionary of cache statistics.
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            # Get the most frequently accessed tables
            top_tables = sorted(
                self._table_access_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                "enabled": self._enabled,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "invalidations": self._invalidations,
                "inserts": self._inserts,
                "top_tables": dict(top_tables),
                "memory_usage_estimate_kb": self._estimate_memory_usage() // 1024
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate the memory usage of the cache.
        
        Returns:
            Estimated memory usage in bytes.
        """
        # This is a rough estimate, not exact
        with self._lock:
            # Count the basic overhead
            total_bytes = sys.getsizeof(self._cache) + sys.getsizeof(self._table_access_count)
            
            # Sample a few entries to get an average size
            sample_size = min(10, len(self._cache))
            if sample_size == 0:
                return total_bytes
            
            sample_bytes = 0
            for i, (key, entry) in enumerate(self._cache.items()):
                if i >= sample_size:
                    break
                
                # Add key size
                sample_bytes += sys.getsizeof(key) + sys.getsizeof(key.query) + sys.getsizeof(key.params)
                
                # Add entry size
                sample_bytes += sys.getsizeof(entry) + sys.getsizeof(entry.query) + sys.getsizeof(entry.params)
                
                # Add result size (this is the biggest part typically)
                sample_bytes += sys.getsizeof(entry.result)
                
                # If result is a list, add size of items
                if isinstance(entry.result, list):
                    for item in entry.result[:5]:  # Sample just a few items
                        sample_bytes += sys.getsizeof(item)
                        if isinstance(item, dict):
                            for k, v in item.items():
                                sample_bytes += sys.getsizeof(k) + sys.getsizeof(v)
            
            # Extrapolate to full cache
            avg_entry_size = sample_bytes / sample_size
            total_bytes += avg_entry_size * len(self._cache)
            
            return int(total_bytes)


# Global cache instance
_GLOBAL_CACHE = QueryCache()


def get_global_cache() -> QueryCache:
    """Get the global query cache instance.
    
    Returns:
        The global query cache.
    """
    return _GLOBAL_CACHE


class QueryTransformer:
    """Utility for transforming and optimizing queries."""
    
    @staticmethod
    def transform_query(query: str, params: Tuple) -> Tuple[str, Tuple]:
        """Transform a query and parameters to optimize for SAP HANA.
        
        This function applies various optimizations to queries before execution.
        
        Args:
            query: The SQL query string.
            params: The query parameters.
            
        Returns:
            The transformed query and parameters.
        """
        # Normalize query for better caching (remove extra whitespace)
        query = ' '.join(query.split())
        
        # Add optimization hints for large result sets
        if "FROM MCTS_TREES" in query and "ORDER BY" in query and "LIMIT" in query:
            # Add a hint to use the query optimizer
            query = query.replace(
                "ORDER BY", 
                "WITH HINT(USE_OLAP_PLAN, IGNORE_PLAN_CACHE) ORDER BY"
            )
        
        # Add hint for complex JSON operations
        if "JSON_VALUE" in query and "JSON_EXISTS" in query:
            # Add hint to parallelize JSON processing
            query = "/*+ PARALLEL(4) */ " + query
        
        # Use materialized views when possible
        query = QueryTransformer.use_materialized_views(query)
        
        return query, params
    
    @staticmethod
    def use_materialized_views(query: str) -> str:
        """Replace tables with materialized views when applicable.
        
        Args:
            query: The SQL query string.
            
        Returns:
            The query with tables replaced by materialized views where appropriate.
        """
        # View mappings: table pattern -> materialized view
        view_mappings = {
            # Tagged trees view
            "FROM MCTX.MCTS_TREES WHERE JSON_EXISTS(metadata, '$.tags')": 
                "FROM MCTX.MV_TAGGED_TREES",
            "FROM MCTS_TREES WHERE JSON_EXISTS(metadata, '$.tags')": 
                "FROM MCTX.MV_TAGGED_TREES",
                
            # GPU trees view
            "FROM MCTX.MCTS_TREES WHERE JSON_VALUE(metadata, '$.gpu_accelerated') = 'true'": 
                "FROM MCTX.MV_GPU_TREES WHERE gpu_accelerated = 'true'",
            "FROM MCTS_TREES WHERE JSON_VALUE(metadata, '$.gpu_accelerated') = 'true'": 
                "FROM MCTX.MV_GPU_TREES WHERE gpu_accelerated = 'true'",
                
            # Performance metrics view
            "FROM MCTX.MCTS_TREES WHERE JSON_EXISTS(metadata, '$.performance')": 
                "FROM MCTX.MV_PERFORMANCE_METRICS",
            "FROM MCTS_TREES WHERE JSON_EXISTS(metadata, '$.performance')": 
                "FROM MCTX.MV_PERFORMANCE_METRICS"
        }
        
        # Apply mappings
        transformed_query = query
        for pattern, replacement in view_mappings.items():
            if pattern in transformed_query:
                transformed_query = transformed_query.replace(pattern, replacement)
        
        return transformed_query
    
    @staticmethod
    def add_query_hints(query: str, hints: List[str]) -> str:
        """Add optimization hints to a query.
        
        Args:
            query: The SQL query string.
            hints: List of hints to add.
            
        Returns:
            The query with added hints.
        """
        if not hints:
            return query
        
        hints_text = " ".join(f"/*+ {hint} */" for hint in hints)
        
        # Add hints after SELECT
        if query.upper().startswith("SELECT"):
            return query.replace("SELECT", f"SELECT {hints_text}", 1)
        
        return query


def query_transform(query: str, params: Tuple) -> Tuple[str, Tuple]:
    """Transform a query and parameters to optimize for SAP HANA.
    
    This function is a compatibility wrapper for QueryTransformer.
    
    Args:
        query: The SQL query string.
        params: The query parameters.
        
    Returns:
        The transformed query and parameters.
    """
    return QueryTransformer.transform_query(query, params)


def cached_query(ttl: Optional[int] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching query results.
    
    Args:
        ttl: Time-to-live in seconds (defaults to cache default).
        
    Returns:
        A decorator function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Extract connection from the first argument
            if not args or not isinstance(args[0], HanaConnection):
                # If no connection is provided, just call the function
                return func(*args, **kwargs)
            
            # Only cache if caching is enabled for the connection
            connection = args[0]
            if not connection.config.enable_caching:
                return func(*args, **kwargs)
            
            # Get the query and parameters from the function arguments
            try:
                # Most query functions have a query and params argument
                # This is a simplified example - actual implementation would need to be adapted
                # to the specific function signatures
                query = args[1] if len(args) > 1 else kwargs.get("query")
                params = args[2] if len(args) > 2 else kwargs.get("params", ())
                
                if not isinstance(params, tuple):
                    params = tuple(params) if params else ()
                
                # Check if the result is in the cache
                cache = get_global_cache()
                cached_result = cache.get(query, params)
                
                if cached_result is not None:
                    logger.debug("Cache hit for query: %s", query)
                    return cached_result
                
                # Apply query transformations
                optimized_query, optimized_params = query_transform(query, params)
                
                # Call the original function with optimized query
                modified_args = list(args)
                if len(modified_args) > 1:
                    modified_args[1] = optimized_query
                if len(modified_args) > 2:
                    modified_args[2] = optimized_params
                
                # Execute the query with a timeout
                start_time = time.time()
                result = func(*modified_args, **kwargs)
                query_time = time.time() - start_time
                
                # Cache the result
                if query_time > 0.1:  # Only cache queries that take some time
                    cache.set(query, params, result, ttl)
                    logger.debug("Cached result for query taking %.2fs: %s", query_time, query)
                
                return result
            except (TypeError, ValueError, AttributeError) as e:
                # If we can't extract the query for any reason, just call the function
                logger.debug(f"Could not extract query for caching, executing without cache: {e}")
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def invalidate_tree_cache(tree_id: Optional[str] = None) -> int:
    """Invalidate cache entries related to MCTS trees.
    
    Args:
        tree_id: Optional specific tree ID to invalidate.
        
    Returns:
        The number of entries invalidated.
    """
    cache = get_global_cache()
    
    if tree_id:
        # Invalidate cache entries for a specific tree
        pattern = f"tree_id = '{tree_id}'"
        return cache.invalidate(query_pattern=pattern)
    else:
        # Invalidate all tree-related cache entries
        return cache.invalidate(table_names=["MCTS_TREES", "MCTS_NODES"])


def invalidate_cache_for_table(table_name: str) -> int:
    """Invalidate cache entries for a specific table.
    
    Args:
        table_name: The name of the table.
        
    Returns:
        The number of entries invalidated.
    """
    cache = get_global_cache()
    return cache.invalidate(table_names=[table_name])


def optimize_and_cache_query(
    query: str, 
    params: Optional[Dict[str, Any]] = None,
    use_views: bool = True,
    add_hints: bool = True,
    ttl: Optional[int] = None
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Optimize a query and check the cache.
    
    Args:
        query: The SQL query to optimize
        params: Query parameters
        use_views: Whether to use materialized views
        add_hints: Whether to add optimization hints
        ttl: Time-to-live for cache entries
        
    Returns:
        Tuple of (optimized query, cached results or None)
    """
    # Convert params to tuple for cache lookup
    param_tuple = tuple() if params is None else tuple(params.items())
    
    # First, check if the result is in the cache
    cache = get_global_cache()
    cached_result = cache.get(query, param_tuple)
    
    if cached_result is not None:
        return query, cached_result
    
    # Apply optimizations
    optimized_query = query
    
    if use_views:
        optimized_query = QueryTransformer.use_materialized_views(optimized_query)
    
    if add_hints:
        hints = ["USE_PARALLEL(4)", "USE_COLUMN_STORE"]
        optimized_query = QueryTransformer.add_query_hints(optimized_query, hints)
    
    # Check if the optimized query is in the cache
    if optimized_query != query:
        cached_result = cache.get(optimized_query, param_tuple)
    
    return optimized_query, cached_result


def cache_query_results(
    query: str,
    params: Optional[Dict[str, Any]],
    results: List[Dict[str, Any]],
    ttl: Optional[int] = None
) -> None:
    """Cache query results.
    
    Args:
        query: The SQL query
        params: Query parameters
        results: Query results to cache
        ttl: Time-to-live for the cache entry
    """
    # Convert params to tuple for cache storage
    param_tuple = tuple() if params is None else tuple(params.items())
    
    # Store in cache
    cache = get_global_cache()
    cache.set(query, param_tuple, results, ttl)


def apply_optimized_query(connection: HanaConnection, 
                        query: str, 
                        params: Dict[str, Any] = None,
                        timeout: int = DEFAULT_QUERY_TIMEOUT,
                        use_cache: bool = True) -> List[Dict[str, Any]]:
    """Execute an optimized query with caching and timeout.
    
    Args:
        connection: The SAP HANA connection.
        query: The SQL query string.
        params: The query parameters.
        timeout: Query timeout in seconds.
        use_cache: Whether to use the query cache.
        
    Returns:
        The query results.
        
    Raises:
        TimeoutError: If the query takes too long to execute.
    """
    # Convert params to tuple for cache lookup
    param_tuple = tuple() if params is None else tuple(params.items())
    
    # Optimize the query and check cache
    optimized_query, cached_results = optimize_and_cache_query(
        query, params, use_views=True, add_hints=True
    )
    
    if use_cache and cached_results is not None:
        return cached_results
    
    # Execute the query
    conn = connection.get_connection()
    if conn is None:
        raise RuntimeError("Could not get database connection")
    
    try:
        # Set query timeout
        cursor = conn.cursor()
        cursor.execute(f"SET STATEMENT_TIMEOUT = {timeout * 1000}")  # Convert to milliseconds
        
        # Execute the query
        start_time = time.time()
        
        # Handle parameters
        if params:
            cursor.execute(optimized_query, params)
        else:
            cursor.execute(optimized_query)
        
        # Get column names
        columns = [desc[0] for desc in cursor.description]
        
        # Fetch results as dictionaries
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        query_time = time.time() - start_time
        
        # Cache the result for slow queries
        if use_cache and query_time > 0.1:
            cache_query_results(query, params, results)
            
            # Log caching for slower queries
            if query_time > 1.0:
                logger.info(f"Cached slow query ({query_time:.2f}s): {query[:100]}...")
        
        return results
    
    except Exception as e:
        # Check if this is a timeout exception
        if "execution timeout" in str(e).lower():
            raise TimeoutError(f"Query timed out after {timeout} seconds: {query}")
        
        # Log other errors
        logger.error(f"Query error: {e}")
        raise
    
    finally:
        # Release the connection
        connection.release_connection(conn)


def optimize_json_path_filtering(json_filters: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Optimize JSON path filtering by grouping filters by path depth.
    
    This function generates efficient SQL conditions for filtering on nested JSON fields.
    
    Args:
        json_filters: List of filter specifications with path, operator and value.
        
    Returns:
        A tuple of SQL WHERE condition and parameters.
    """
    if not json_filters:
        return "", {}
    
    # Group filters by path depth
    path_groups = {}
    for i, filter_spec in enumerate(json_filters):
        path = filter_spec["path"]
        path_parts = path.split(".")
        depth = len(path_parts)
        
        if depth not in path_groups:
            path_groups[depth] = []
        
        path_groups[depth].append((i, filter_spec))
    
    conditions = []
    params = {}
    
    # Process each depth group
    for depth in sorted(path_groups.keys()):
        filters = path_groups[depth]
        
        if depth == 1:
            # Simple paths can use direct JSON_VALUE
            for i, filter_spec in filters:
                path = filter_spec["path"]
                operator = filter_spec["operator"]
                value = filter_spec["value"]
                
                # Parameter name
                param_name = f"json_val_{i}"
                
                # Build condition based on operator
                if operator == "eq":
                    condition = f"JSON_VALUE(metadata, '$.{path}') = :{param_name}"
                elif operator == "neq":
                    condition = f"JSON_VALUE(metadata, '$.{path}') != :{param_name}"
                elif operator == "contains":
                    condition = f"JSON_VALUE(metadata, '$.{path}') LIKE :{param_name}"
                    value = f"%{value}%"
                elif operator == "starts_with":
                    condition = f"JSON_VALUE(metadata, '$.{path}') LIKE :{param_name}"
                    value = f"{value}%"
                elif operator == "ends_with":
                    condition = f"JSON_VALUE(metadata, '$.{path}') LIKE :{param_name}"
                    value = f"%{value}"
                elif operator == "in":
                    if isinstance(value, list):
                        in_conditions = []
                        for j, val in enumerate(value):
                            in_param_name = f"{param_name}_{j}"
                            in_conditions.append(f":{in_param_name}")
                            params[in_param_name] = val
                        condition = f"JSON_VALUE(metadata, '$.{path}') IN ({', '.join(in_conditions)})"
                        continue  # Skip adding the main parameter
                    else:
                        condition = f"JSON_VALUE(metadata, '$.{path}') = :{param_name}"
                else:
                    # Default to equality
                    condition = f"JSON_VALUE(metadata, '$.{path}') = :{param_name}"
                
                conditions.append(condition)
                params[param_name] = value
        else:
            # For nested paths, use JSON_TABLE for better performance
            # Extract the common parent path
            parent_paths = set()
            for _, filter_spec in filters:
                path_parts = filter_spec["path"].split(".")
                parent_path = ".".join(path_parts[:-1])
                parent_paths.add(parent_path)
            
            # Process each parent path
            for parent_path in parent_paths:
                parent_filters = [
                    (i, f) for i, f in filters 
                    if f["path"].startswith(f"{parent_path}.")
                ]
                
                if parent_filters:
                    # Use JSON_TABLE to extract all fields under this parent
                    # This is more efficient than multiple JSON_VALUE calls
                    table_alias = f"jt_{parent_path.replace('.', '_')}"
                    json_table_columns = []
                    table_conditions = []
                    
                    for i, filter_spec in parent_filters:
                        path = filter_spec["path"]
                        leaf_name = path.split(".")[-1]
                        column_alias = f"col_{i}"
                        
                        # Add column to JSON_TABLE
                        json_table_columns.append(
                            f"'$.{leaf_name}' AS {column_alias}"
                        )
                        
                        # Add condition on the extracted column
                        operator = filter_spec["operator"]
                        value = filter_spec["value"]
                        param_name = f"json_val_{i}"
                        
                        # Build condition based on operator
                        if operator == "eq":
                            condition = f"{table_alias}.{column_alias} = :{param_name}"
                        elif operator == "neq":
                            condition = f"{table_alias}.{column_alias} != :{param_name}"
                        elif operator == "contains":
                            condition = f"{table_alias}.{column_alias} LIKE :{param_name}"
                            value = f"%{value}%"
                        elif operator == "starts_with":
                            condition = f"{table_alias}.{column_alias} LIKE :{param_name}"
                            value = f"{value}%"
                        elif operator == "ends_with":
                            condition = f"{table_alias}.{column_alias} LIKE :{param_name}"
                            value = f"%{value}"
                        elif operator == "in":
                            if isinstance(value, list):
                                in_conditions = []
                                for j, val in enumerate(value):
                                    in_param_name = f"{param_name}_{j}"
                                    in_conditions.append(f":{in_param_name}")
                                    params[in_param_name] = val
                                condition = f"{table_alias}.{column_alias} IN ({', '.join(in_conditions)})"
                                table_conditions.append(condition)
                                continue  # Skip adding the main parameter
                            else:
                                condition = f"{table_alias}.{column_alias} = :{param_name}"
                        else:
                            # Default to equality
                            condition = f"{table_alias}.{column_alias} = :{param_name}"
                        
                        table_conditions.append(condition)
                        params[param_name] = value
                    
                    # Create JSON_TABLE condition
                    json_table_condition = (
                        f"EXISTS (SELECT 1 FROM JSON_TABLE(metadata, '$.{parent_path}' "
                        f"COLUMNS ({', '.join(json_table_columns)})) AS {table_alias} "
                        f"WHERE {' AND '.join(table_conditions)})"
                    )
                    
                    conditions.append(json_table_condition)
    
    # Combine all conditions with AND
    if conditions:
        return f"({' AND '.join(conditions)})", params
    else:
        return "", {}


def optimize_tree_queries(connection: HanaConnection) -> Dict[str, Any]:
    """Perform optimization of tree queries by precomputing common aggregates.
    
    This function precomputes common aggregates and stores them in cache
    to speed up future queries.
    
    Args:
        connection: The SAP HANA connection.
        
    Returns:
        Statistics about the optimization.
    """
    stats = {
        "precomputed_queries": 0,
        "query_times": {}
    }
    
    # List of common queries to precompute
    common_queries = [
        # Count of trees by creation date (last 30 days)
        (
            f"SELECT DATE_TRUNC('DAY', created_at) as day, COUNT(*) as count "
            f"FROM {connection.config.schema}.MCTS_TREES "
            f"WHERE created_at >= ADD_DAYS(CURRENT_DATE, -30) "
            f"GROUP BY DATE_TRUNC('DAY', created_at) "
            f"ORDER BY day",
            {}
        ),
        
        # Count by batch size ranges
        (
            f"SELECT "
            f"  CASE "
            f"    WHEN batch_size <= 16 THEN 'small' "
            f"    WHEN batch_size <= 64 THEN 'medium' "
            f"    ELSE 'large' "
            f"  END as size_category, "
            f"  COUNT(*) as count "
            f"FROM {connection.config.schema}.MCTS_TREES "
            f"GROUP BY size_category",
            {}
        ),
        
        # GPU vs CPU distribution
        (
            f"SELECT "
            f"  CASE "
            f"    WHEN JSON_VALUE(metadata, '$.gpu_accelerated') = 'true' THEN 'gpu' "
            f"    WHEN JSON_VALUE(metadata, '$.gpu_serialized') = 'true' THEN 'gpu' "
            f"    WHEN JSON_VALUE(metadata, '$.environment.platform') = 'gpu' THEN 'gpu' "
            f"    ELSE 'cpu' "
            f"  END as platform, "
            f"  COUNT(*) as count "
            f"FROM {connection.config.schema}.MCTS_TREES "
            f"GROUP BY platform",
            {}
        ),
    ]
    
    # Precompute these queries
    for query, params in common_queries:
        try:
            start_time = time.time()
            results = apply_optimized_query(
                connection, query, params, use_cache=False
            )
            query_time = time.time() - start_time
            
            # Cache the results with a longer TTL
            cache_query_results(query, params, results, ttl=3600)  # 1 hour
            
            stats["precomputed_queries"] += 1
            stats["query_times"][query[:50]] = query_time
            
        except Exception as e:
            logger.error(f"Error precomputing query: {e}")
    
    # Precompute some aggregates for common filters
    try:
        # Average simulation counts
        query = (
            f"SELECT AVG(num_simulations) as avg_simulations "
            f"FROM {connection.config.schema}.MCTS_TREES"
        )
        results = apply_optimized_query(connection, query, use_cache=False)
        cache_query_results(query, None, results, ttl=3600)
        
        stats["precomputed_queries"] += 1
    except Exception as e:
        logger.error(f"Error precomputing aggregates: {e}")
    
    return stats


# Import sys for memory usage estimation
import sys

# Global query cache instance with proper initialization
global_query_cache = get_global_cache()