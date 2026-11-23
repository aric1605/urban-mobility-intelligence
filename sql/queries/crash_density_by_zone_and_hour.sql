-- Crash density by zone with casualty aggregates and ranking
-- Parameters: :min_crashes

WITH crash_stats AS (
    SELECT 
        c.zone_id,
        z.zone_name,
        z.borough,
        COUNT(c.crash_id) AS total_crashes,
        SUM(c.injured) AS total_injured,
        SUM(c.killed) AS total_killed
    FROM crashes c
    JOIN zones z ON c.zone_id = z.zone_id
    WHERE c.zone_id > 0
    GROUP BY c.zone_id, z.zone_name, z.borough
    HAVING COUNT(c.crash_id) >= :min_crashes
)
SELECT 
    DENSE_RANK() OVER (ORDER BY total_crashes DESC) AS crash_rank,
    zone_id,
    zone_name,
    borough,
    total_crashes,
    total_injured,
    total_killed
FROM crash_stats
ORDER BY total_crashes DESC;
