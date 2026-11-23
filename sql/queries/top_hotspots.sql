-- Rank zones by mean congestion_index in a selected hour range with a minimum-trips filter
-- Parameters: :start_hour, :end_hour, :min_trips

WITH zone_metrics AS (
    SELECT 
        zh.zone_id,
        z.zone_name,
        z.borough,
        COUNT(zh.hour_ts) AS active_hours,
        SUM(zh.trips) AS total_trips,
        ROUND(AVG(zh.avg_speed), 2) AS mean_speed_mph,
        ROUND(AVG(zh.congestion_index), 4) AS mean_congestion
    FROM zone_hourly zh
    JOIN zones z ON zh.zone_id = z.zone_id
    WHERE CAST(strftime('%H', zh.hour_ts) AS INTEGER) BETWEEN :start_hour AND :end_hour
      AND zh.congestion_index IS NOT NULL
    GROUP BY zh.zone_id, z.zone_name, z.borough
    HAVING SUM(zh.trips) >= :min_trips
)
SELECT 
    RANK() OVER (ORDER BY mean_congestion DESC) AS congestion_rank,
    zone_id,
    zone_name,
    borough,
    active_hours,
    total_trips,
    mean_speed_mph,
    mean_congestion
FROM zone_metrics
ORDER BY congestion_rank;
