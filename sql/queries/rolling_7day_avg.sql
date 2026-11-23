-- 7-day (168-hour) rolling average of city congestion using a window frame
-- Parameters: None

WITH city_ordered AS (
    SELECT 
        hour_ts,
        trips,
        avg_speed,
        congestion_index
    FROM city_hourly
    ORDER BY hour_ts
)
SELECT 
    hour_ts,
    trips,
    avg_speed,
    congestion_index,
    ROUND(AVG(congestion_index) OVER (
        ORDER BY hour_ts 
        ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
    ), 4) AS rolling_7day_congestion,
    ROUND(AVG(avg_speed) OVER (
        ORDER BY hour_ts 
        ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_7day_speed
FROM city_ordered
ORDER BY hour_ts;
