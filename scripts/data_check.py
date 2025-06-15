import psycopg2
import os
import json
from datetime import datetime, timedelta

# Database connection parameters
db_config = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USERNAME"],
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"]
}

def check_for_new_data(hours_back=24):
    """Check if there are new or updated records in the last hours_back hours"""
    try:
        # Connect to the database
        with psycopg2.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                # Get timestamp for comparison
                time_threshold = datetime.now() - timedelta(hours=hours_back)
                
                # Check for new or updated records in page_raw
                cursor.execute("""
                    SELECT COUNT(*) FROM page_raw 
                    WHERE last_scraped > %s AND is_active = TRUE
                """, (time_threshold,))
                raw_count = cursor.fetchone()[0]
                
                # Check for new or updated records in page_content that are not in page_keywords
                cursor.execute("""
                    SELECT COUNT(*) FROM page_content pc
                    LEFT JOIN page_keywords pk ON pc.id = pk.id
                    WHERE pc.last_scraped > %s 
                    AND pc.is_active = TRUE 
                    AND (pk.id IS NULL OR pc.last_updated > pk.last_modified)
                """, (time_threshold,))
                keywords_needed = cursor.fetchone()[0]
                
                # Check for new or updated records in page_content that are not in page_embeddings_a
                cursor.execute("""
                    SELECT COUNT(*) FROM page_content pc
                    LEFT JOIN (
                        SELECT DISTINCT id FROM page_embeddings_a
                    ) pe ON pc.id = pe.id
                    WHERE pc.last_scraped > %s 
                    AND pc.is_active = TRUE 
                    AND (pe.id IS NULL)
                """, (time_threshold,))
                embeddings_needed = cursor.fetchone()[0]
                
                # Return results
                return {
                    "new_or_updated_raw": raw_count,
                    "keywords_needed": keywords_needed,
                    "embeddings_needed": embeddings_needed,
                    "processing_needed": keywords_needed > 0 or embeddings_needed > 0
                }
    except Exception as e:
        print(f"Error checking for new data: {e}")
        # Default to processing needed in case of error
        return {
            "error": str(e),
            "processing_needed": True
        }

# Run the check function and output results
results = check_for_new_data()
print(f"Check results: {results}")

# Set GitHub output variable
if 'GITHUB_OUTPUT' in os.environ:
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f"processing_needed={str(results['processing_needed']).lower()}\n")
        f.write(f"keywords_needed={results.get('keywords_needed', 0)}\n")
        f.write(f"embeddings_needed={results.get('embeddings_needed', 0)}\n")

# Save results to a JSON file
with open('data_check_results.json', 'w') as f:
    json.dump(results, f, indent=2) 