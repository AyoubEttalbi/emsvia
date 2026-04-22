import { useState, useEffect, useCallback } from 'react';
import { getPendingUnknowns } from '../api/client';

export function useUnknowns() {
    const [unknowns, setUnknowns] = useState([]);
    const [loading, setLoading] = useState(true);

    const fetch = useCallback(async () => {
        try {
            const response = await getPendingUnknowns();
            setUnknowns(response.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetch();
        const interval = setInterval(fetch, 15000); // 15s
        return () => clearInterval(interval);
    }, [fetch]);

    return { unknowns, loading, refetch: fetch };
}
