import { useState, useEffect } from 'react';
import { getRecentAttendance } from '../api/client';

export function useActivity() {
    const [activities, setActivities] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetch = async () => {
            try {
                const response = await getRecentAttendance();
                setActivities(response.data);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetch();
        const interval = setInterval(fetch, 10000); // 10s
        return () => clearInterval(interval);
    }, []);

    return { activities, loading };
}
