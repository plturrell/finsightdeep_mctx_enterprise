-- SAP HANA Performance Optimizations for MCTX
-- This script creates optimized indexes, materialized views, and statistics
-- for improved query performance with large MCTX tree datasets

-- Schema creation (if not exists)
CREATE SCHEMA IF NOT EXISTS MCTX;

-- Enable logging for optimization operations
CREATE TABLE IF NOT EXISTS MCTX.OPTIMIZATION_LOG (
    timestamp TIMESTAMP,
    operation VARCHAR(255),
    details VARCHAR(1000)
);

-- Record the start of optimization
INSERT INTO MCTX.OPTIMIZATION_LOG VALUES (CURRENT_TIMESTAMP, 'START_OPTIMIZATION', 'Starting enhanced database optimization process');

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Create indexes on frequently filtered columns
CREATE INDEX IF NOT EXISTS IDX_TREES_NAME ON MCTX.MCTS_TREES(name);
CREATE INDEX IF NOT EXISTS IDX_TREES_CREATED_AT ON MCTX.MCTS_TREES(created_at);
CREATE INDEX IF NOT EXISTS IDX_TREES_UPDATED_AT ON MCTX.MCTS_TREES(updated_at);
CREATE INDEX IF NOT EXISTS IDX_TREES_BATCH_SIZE ON MCTX.MCTS_TREES(batch_size);
CREATE INDEX IF NOT EXISTS IDX_TREES_NUM_ACTIONS ON MCTX.MCTS_TREES(num_actions);
CREATE INDEX IF NOT EXISTS IDX_TREES_NUM_SIMULATIONS ON MCTX.MCTS_TREES(num_simulations);

-- Composite index for range queries (significantly improves multi-condition filters)
CREATE INDEX IF NOT EXISTS IDX_TREES_DIMENSIONS ON MCTX.MCTS_TREES(batch_size, num_actions, num_simulations);

-- JSON Metadata Indexes
-- These significantly improve filtering on metadata fields

-- Index for common metadata fields
CREATE INDEX IF NOT EXISTS IDX_TREES_METADATA_VERSION ON MCTX.MCTS_TREES(
    JSON_VALUE(metadata, '$.model_version')
);

-- Index for GPU acceleration flag
CREATE INDEX IF NOT EXISTS IDX_TREES_GPU_ACCELERATED ON MCTX.MCTS_TREES(
    JSON_VALUE(metadata, '$.gpu_accelerated')
);

-- Index for frequently used tags
CREATE INDEX IF NOT EXISTS IDX_TREES_TAGS ON MCTX.MCTS_TREES(
    JSON_VALUE(metadata, '$.tags')
);

-- Performance-related indexes
CREATE INDEX IF NOT EXISTS IDX_TREES_VALIDATION_ACCURACY ON MCTX.MCTS_TREES(
    JSON_VALUE(metadata, '$.performance.validation_accuracy')
);

CREATE INDEX IF NOT EXISTS IDX_TREES_TRAINING_ACCURACY ON MCTX.MCTS_TREES(
    JSON_VALUE(metadata, '$.performance.training_accuracy')
);

CREATE INDEX IF NOT EXISTS IDX_TREES_PLATFORM ON MCTX.MCTS_TREES(
    JSON_VALUE(metadata, '$.environment.platform')
);

-- Performance indexes for tree nodes (improve tree traversal queries)
CREATE INDEX IF NOT EXISTS IDX_NODES_TREE_ID ON MCTX.MCTS_NODES(tree_id);
CREATE INDEX IF NOT EXISTS IDX_NODES_PARENT_ID ON MCTX.MCTS_NODES(parent_id);
CREATE INDEX IF NOT EXISTS IDX_NODES_VISIT_COUNT ON MCTX.MCTS_NODES(visit_count);
CREATE INDEX IF NOT EXISTS IDX_NODES_VALUE ON MCTX.MCTS_NODES(value);

-- Create Full-Text Indexes for search
CREATE FULLTEXT INDEX IF NOT EXISTS FTI_TREES_METADATA ON MCTX.MCTS_TREES(metadata)
FAST PREPROCESS OFF
ASYNC
LANGUAGE DETECTION ('en');

CREATE FULLTEXT INDEX IF NOT EXISTS FTI_TREES_NAME ON MCTX.MCTS_TREES(name)
FAST PREPROCESS OFF
ASYNC;

-- ============================================================================
-- MATERIALIZED VIEWS
-- ============================================================================

-- Materialized view for tagged trees (frequently accessed)
CREATE MATERIALIZED VIEW IF NOT EXISTS MCTX.MV_TAGGED_TREES
REFRESH FAST ON COMMIT
AS 
SELECT 
    tree_id,
    name,
    created_at,
    updated_at,
    batch_size,
    num_actions,
    num_simulations,
    JSON_VALUE(metadata, '$.tags[0]') as tag1,
    JSON_VALUE(metadata, '$.tags[1]') as tag2,
    JSON_VALUE(metadata, '$.tags[2]') as tag3,
    JSON_VALUE(metadata, '$.tags[3]') as tag4,
    JSON_VALUE(metadata, '$.tags[4]') as tag5,
    JSON_VALUE(metadata, '$.model_version') as model_version,
    JSON_VALUE(metadata, '$.description') as description,
    JSON_VALUE(metadata, '$.gpu_accelerated') as gpu_accelerated
FROM MCTX.MCTS_TREES
WHERE JSON_EXISTS(metadata, '$.tags');

-- Materialized view for GPU accelerated trees
CREATE MATERIALIZED VIEW IF NOT EXISTS MCTX.MV_GPU_TREES
REFRESH FAST ON COMMIT
AS 
SELECT 
    tree_id,
    name,
    created_at,
    updated_at,
    batch_size,
    num_actions,
    num_simulations,
    JSON_VALUE(metadata, '$.environment.platform') as platform,
    JSON_VALUE(metadata, '$.environment.gpu_model') as gpu_model,
    JSON_VALUE(metadata, '$.tensor_cores_used') as tensor_cores_used
FROM MCTX.MCTS_TREES
WHERE 
    JSON_VALUE(metadata, '$.gpu_accelerated') = 'true' OR 
    JSON_VALUE(metadata, '$.gpu_serialized') = 'true' OR
    JSON_VALUE(metadata, '$.environment.platform') = 'gpu';

-- Materialized view for performance metrics (frequently accessed)
CREATE MATERIALIZED VIEW IF NOT EXISTS MCTX.MV_PERFORMANCE_METRICS
REFRESH FAST
EVERY 60 MINUTES
AS
SELECT 
    tree_id,
    name,
    created_at,
    batch_size,
    num_actions,
    num_simulations,
    CAST(JSON_VALUE(metadata, '$.performance.training_accuracy') AS DECIMAL(10,5)) as training_accuracy,
    CAST(JSON_VALUE(metadata, '$.performance.validation_accuracy') AS DECIMAL(10,5)) as validation_accuracy,
    CAST(JSON_VALUE(metadata, '$.performance.inference_time_ms') AS DECIMAL(10,2)) as inference_time_ms,
    JSON_VALUE(metadata, '$.environment.platform') as platform
FROM MCTX.MCTS_TREES
WHERE JSON_EXISTS(metadata, '$.performance');

-- Materialized view for common tree statistics (frequently accessed)
CREATE MATERIALIZED VIEW IF NOT EXISTS MCTX.MV_TREE_STATS
REFRESH FAST
EVERY 60 MINUTES
AS
SELECT 
    t.tree_id,
    t.name,
    t.created_at,
    t.batch_size,
    t.num_actions,
    t.num_simulations,
    COUNT(n.id) as node_count,
    AVG(n.visit_count) as avg_visit_count,
    MAX(n.visit_count) as max_visit_count,
    AVG(n.value) as avg_value
FROM 
    MCTX.MCTS_TREES t
JOIN 
    MCTX.MCTS_NODES n ON t.tree_id = n.tree_id
GROUP BY 
    t.tree_id, t.name, t.created_at, t.batch_size, t.num_actions, t.num_simulations;

-- ============================================================================
-- STATISTICS
-- ============================================================================

-- Update statistics for the tables and indexes
MERGE STATISTICS ON MCTX.MCTS_TREES;
MERGE STATISTICS ON MCTX.MCTS_NODES;
MERGE STATISTICS ON MCTX.MV_TAGGED_TREES;
MERGE STATISTICS ON MCTX.MV_GPU_TREES;
MERGE STATISTICS ON MCTX.MV_PERFORMANCE_METRICS;
MERGE STATISTICS ON MCTX.MV_TREE_STATS;

-- ============================================================================
-- COLUMN STORE OPTIMIZATION
-- ============================================================================

-- Ensure tables are using column store when appropriate
ALTER TABLE MCTX.MCTS_TREES WITH PARAMETERS ('COLUMN');
ALTER TABLE MCTX.MCTS_NODES WITH PARAMETERS ('COLUMN');

-- ============================================================================
-- PARTITIONING
-- ============================================================================

-- Partition large tables by created_at for better performance
-- This is especially useful for time-series queries
-- Uncomment when the dataset is large enough to benefit from partitioning
/*
ALTER TABLE MCTX.MCTS_TREES ADD PARTITION BY RANGE(created_at)
(
    PARTITION p_2020 <= '2020-12-31',
    PARTITION p_2021 <= '2021-12-31',
    PARTITION p_2022 <= '2022-12-31',
    PARTITION p_2023 <= '2023-12-31',
    PARTITION p_2024 <= '2024-12-31',
    PARTITION p_others
);
*/

-- ============================================================================
-- DATABASE PARAMETERS
-- ============================================================================

-- Set optimal parameters for large dataset processing
-- Note: These should be adjusted based on the specific environment
ALTER SYSTEM ALTER CONFIGURATION ('indexserver.ini', 'SYSTEM') 
SET ('parallel', 'max_concurrency_hint') = '16';

ALTER SYSTEM ALTER CONFIGURATION ('global.ini', 'SYSTEM') 
SET ('memorymanager', 'statement_memory_limit') = '2048';

-- ============================================================================
-- QUERY EXECUTION PLAN HINTS
-- ============================================================================

-- Create procedure with hints for optimal tree retrieval
CREATE OR REPLACE PROCEDURE MCTX.GET_OPTIMIZED_TREES(
    IN p_batch_size INTEGER,
    IN p_min_depth INTEGER,
    IN p_tag VARCHAR(100)
)
LANGUAGE SQLSCRIPT
SQL SECURITY INVOKER
AS
BEGIN
    /*+ USE_PARALLEL(4) */
    /*+ USE_COLUMN_STORE */
    SELECT 
        t.tree_id,
        t.name,
        t.created_at,
        t.batch_size,
        t.num_actions,
        t.num_simulations,
        t.metadata
    FROM 
        MCTX.MCTS_TREES t
    WHERE 
        t.batch_size >= :p_batch_size AND
        JSON_VALUE(t.metadata, '$.depth') >= :p_min_depth AND
        JSON_VALUE(t.metadata, '$.tags') LIKE '%' || :p_tag || '%'
    ORDER BY 
        t.created_at DESC;
END;

-- Create procedure with hints for efficient node retrieval
CREATE OR REPLACE PROCEDURE MCTX.GET_OPTIMIZED_NODES(
    IN p_tree_id VARCHAR(36),
    IN p_min_visits INTEGER
)
LANGUAGE SQLSCRIPT
SQL SECURITY INVOKER
AS
BEGIN
    /*+ USE_PARALLEL(4) */
    /*+ USE_COLUMN_STORE */
    SELECT 
        n.id,
        n.tree_id,
        n.parent_id,
        n.visit_count,
        n.value,
        n.state,
        n.action
    FROM 
        MCTX.MCTS_NODES n
    WHERE 
        n.tree_id = :p_tree_id AND
        n.visit_count >= :p_min_visits
    ORDER BY 
        n.visit_count DESC;
END;

-- Log completion
INSERT INTO MCTX.OPTIMIZATION_LOG VALUES (CURRENT_TIMESTAMP, 'COMPLETE_OPTIMIZATION', 'Enhanced database optimization process completed');

-- Provide optimization summary
SELECT 'Enhanced Optimization Complete' as status, 
       (SELECT COUNT(*) FROM MCTX.MCTS_TREES) as total_trees,
       CURRENT_TIMESTAMP as timestamp;