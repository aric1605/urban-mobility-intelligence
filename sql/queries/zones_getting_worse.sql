-- Compare first-half vs second-half average congestion per zone
-- Parameters: :midpoint_ts, :min_trips

WITH zone_halves AS (
    SELECT 
        zh.zone_id,
        z.zone_name,
        z.borough,
        CASE 
            WHEN zh.hour_ts < :midpoint_ts THEN 'H1'
            ELSE 'H2'
        END AS half_period,
        COUNT(zh.hour_ts) AS period_hours,
        SUM(zh.trips) AS period_trips,
        AVG(zh.congestion_index) AS mean_congestion
    FROM zone_hourly zh
    JOIN zones z ON zh.zone_id = z.zone_id
    WHERE zh.congestion_index IS NOT NULL
    GROUP BY zh.zone_id, z.zone_name, z.borough, half_period
),
pivoted AS (
    SELECT 
        h1.zone_id,
        h1.zone_name,
        h1.borough,
        ROUND(h1.mean_congestion, 4) AS h1_congestion,
        ROUND(h2.mean_congestion, 4) AS h2_congestion,
        ROUND(h2.mean_congestion - h1.mean_congestion, 4) AS congestion_change,
        h1.period_trips + h2.period_trips AS total_trips
    FROM zone_halves h1
    JOIN zone_halves h2 ON h1.zone_id = h2.zone_id AND h1.half_period = 'H1' AND h2.half_period = 'H2'
    WHERE (h1.period_trips + h2.period_trips) >= :min_trips
)
SELECT 
    RANK() OVER (ORDER BY congestion_change DESC) AS change_rank,
    zone_id,
    zone_name,
    borough,
    h1_congestion,
    h2_congestion,
    congestion_change,
    total_trips,
    CASE 
        WHEN congestion_change > 0.02 THEN 'Worsening'
        WHEN congestion_change < -0.02 THEN 'Improving'
        ELSE 'Stable'
    END AS status
FROM pivoted
ORDER BY congestion_change DESC;
