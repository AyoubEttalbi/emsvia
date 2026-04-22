import { useState, useEffect, useCallback } from 'react';
import { getHealth } from '../api/client';

export function useHealth() {
    const [health, setHealth] = useState(null);
    const [loading, setLoading] = useState(true);

    const fetch = useCallback(async () => {
        try {
            const response = await getHealth();
            setHealth(response.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetch();
        const interval = setInterval(fetch, 5000);
        return () => clearInterval(interval);
    }, [fetch]);

    return { health, loading, refetch: fetch };
}
