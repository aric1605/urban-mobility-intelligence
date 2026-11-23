-- Hour-of-day x day-of-week mean congestion and worst hour analysis
-- Parameters: :zone_id (-1 for city-wide / all zones)

WITH hourly_breakdown AS (
    SELECT 
        CAST(strftime('%w', hour_ts) AS INTEGER) AS day_of_week,
        CAST(strftime('%H', hour_ts) AS INTEGER) AS hour_of_day,
        ROUND(AVG(congestion_index), 4) AS mean_congestion,
        ROUND(AVG(avg_speed), 2) AS mean_speed,
        SUM(trips) AS total_trips
    FROM zone_hourly
    WHERE (:zone_id = -1 OR zone_id = :zone_id)
      AND congestion_index IS NOT NULL
    GROUP BY day_of_week, hour_of_day
),
ranked_hours AS (
    SELECT 
        day_of_week,
        hour_of_day,
        mean_congestion,
        mean_speed,
        total_trips,
        ROW_NUMBER() OVER (ORDER BY mean_congestion DESC) AS worst_rank,
        LAG(mean_congestion, 1) OVER (PARTITION BY day_of_week ORDER BY hour_of_day) AS prev_hour_congestion
    FROM hourly_breakdown
)
SELECT 
    day_of_week,
    hour_of_day,
    mean_congestion,
    mean_speed,
    total_trips,
    worst_rank,
    prev_hour_congestion,
    ROUND(mean_congestion - COALESCE(prev_hour_congestion, mean_congestion), 4) AS congestion_delta_from_prev_hour
FROM ranked_hours
ORDER BY day_of_week, hour_of_day;
