import asyncio
import time
import httpx
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------
CONCURRENCY_LIMIT = 4
MAX_RETRIES = 3
INITIAL_BACKOFF = 0.5
REQUEST_TIMEOUT = 5.0

SEMAPHORE = asyncio.Semaphore(CONCURRENCY_LIMIT)


async def fetch_post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    item_id: int
) -> Dict[str, Any]:
    """
    Fetches a single post using semaphore concurrency gating and exponential backoff.
    """
    async with SEMAPHORE:
        delay = INITIAL_BACKOFF
        start_time = time.perf_counter()
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # 1. Use the shared client with proper await and timeout
                response = await client.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                
                # 2. Parse payload and compute latency
                data = response.json()
                latency = (time.perf_counter() - start_time) * 1000.0  # ms
                
                return {
                    "item_id": item_id,
                    "status": "SUCCESS",
                    "title": data.get("title", ""),
                    "attempts": attempt,
                    "latency_ms": round(latency, 2)
                }

            except (httpx.HTTPStatusError, httpx.RequestError) as err:
                if attempt == MAX_RETRIES:
                    latency = (time.perf_counter() - start_time) * 1000.0
                    return {
                        "item_id": item_id,
                        "status": "FAILED",
                        "error": str(err),
                        "attempts": attempt,
                        "latency_ms": round(latency, 2)
                    }
                
                # 3. Non-blocking exponential backoff
                await asyncio.sleep(delay)
                delay *= 2

        return {
            "item_id": item_id,
            "status": "FAILED",
            "error": "Exhausted retries without response",
            "attempts": MAX_RETRIES,
            "latency_ms": 0.0
        }


async def main():
    # Construct task list: 18 valid + 2 failure endpoints
    valid_urls = [(i, f"https://jsonplaceholder.typicode.com/posts/{i}") for i in range(1, 19)]
    invalid_urls = [
        (99, "https://jsonplaceholder.typicode.com/posts/999999"),  # Triggers 404 HTTPStatusError
        (100, "https://invalid-non-existent-domain-xyz.com/data")     # Triggers RequestError (DNS)
    ]
    all_targets = valid_urls + invalid_urls

    print(f"[*] Starting ingestion pipeline for {len(all_targets)} targets...")
    print(f"[*] Active Concurrency Cap: {CONCURRENCY_LIMIT} workers")
    
    total_start = time.perf_counter()

    # Shared connection pool for all 20 requests
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_post_with_retry(client, url, item_id)
            for item_id, url in all_targets
        ]
        # Dispatch concurrently with error resilience
        results: List[Dict[str, Any]] = await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = time.perf_counter() - total_start

    # -----------------------------------------------------------------------
    # Metrics & Dashboard Output
    # -----------------------------------------------------------------------
    successes = [r for r in results if isinstance(r, dict) and r.get("status") == "SUCCESS"]
    failures = [r for r in results if isinstance(r, dict) and r.get("status") == "FAILED"]
    
    avg_latency = (
        sum(r["latency_ms"] for r in successes) / len(successes)
        if successes else 0.0
    )

    print("\n" + "=" * 60)
    print("           PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Items Processed : {len(results)}")
    print(f"Successful Ingestions : {len(successes)}")
    print(f"Failed Ingestions     : {len(failures)}")
    print(f"Average Latency (OK)  : {avg_latency:.2f} ms")
    print(f"Total Wall Clock Time : {total_elapsed:.2f} s")
    print("=" * 60)

    print("\n--- Failure Breakdown ---")
    for f in failures:
        print(f"Item #{f['item_id']} | Attempts: {f['attempts']} | Error: {f['error']}")


if __name__ == "__main__":
    asyncio.run(main())
