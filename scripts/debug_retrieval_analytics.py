#!/usr/bin/env python3
"""
Debug script to check retrieval analytics updates.
"""
import sys
import os
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hubert.db.postgres_storage import PostgresStorage


def check_recent_retrieval_analytics():
    """Check recent retrieval analytics records."""
    storage = PostgresStorage()
    
    print("=== Recent Retrieval Analytics ===\n")
    
    # Get recent retrieval analytics with query info
    query = """
    SELECT ra.id, ra.query_analytics_id, qa.query, qa.session_id, 
           ra.retrieved_url, ra.rank_position, ra.similarity_score, 
           ra.is_relevant, ra.timestamp
    FROM retrieval_analytics ra
    JOIN query_analytics qa ON ra.query_analytics_id = qa.id
    WHERE ra.timestamp >= NOW() - INTERVAL '1 hour'
    ORDER BY ra.timestamp DESC
    LIMIT 20;
    """
    
    results = storage._execute_query(query, fetch='all')
    
    if not results:
        print("No recent retrieval analytics records found.")
        return
    
    print(f"Found {len(results)} recent records:\n")
    
    for row in results:
        ra_id, qa_id, query_text, session_id, url, rank, sim_score, is_relevant, timestamp = row
        print(f"RA ID: {ra_id}")
        print(f"  Query Analytics ID: {qa_id}")
        print(f"  Query: '{query_text}'")
        print(f"  Session: {session_id[:30]}...")
        print(f"  URL: {url}")
        print(f"  Rank: {rank}")
        print(f"  Similarity Score: {sim_score}")
        print(f"  Is Relevant: {is_relevant}")
        print(f"  Timestamp: {timestamp}")
        print()


def test_update_query():
    """Test the update query manually."""
    storage = PostgresStorage()
    
    print("=== Testing Update Query ===\n")
    
    # First, find a recent record to update
    query = """
    SELECT ra.id, ra.query_analytics_id, ra.retrieved_url, ra.rank_position, ra.is_relevant
    FROM retrieval_analytics ra
    WHERE ra.timestamp >= NOW() - INTERVAL '1 hour'
    AND ra.is_relevant IS NULL
    LIMIT 1;
    """
    
    result = storage._execute_query(query, fetch='one')
    
    if not result:
        print("No recent records found to test update.")
        return
    
    ra_id, qa_id, url, rank, current_relevant = result
    print(f"Found record to test:")
    print(f"  RA ID: {ra_id}")
    print(f"  Query Analytics ID: {qa_id}")
    print(f"  URL: {url}")
    print(f"  Rank: {rank}")
    print(f"  Current is_relevant: {current_relevant}")
    
    # Try to update it
    print(f"\nTesting update...")
    try:
        # Test the update query
        update_query = """
        UPDATE retrieval_analytics 
        SET is_relevant = %s, timestamp = CURRENT_TIMESTAMP
        WHERE query_analytics_id = %s AND retrieved_url = %s AND rank_position = %s
        RETURNING id, is_relevant;
        """
        
        update_result = storage._execute_query(
            update_query, 
            (True, qa_id, url, rank), 
            fetch='one',
            commit=True
        )
        
        if update_result:
            print(f"Update successful: RA ID {update_result[0]}, is_relevant = {update_result[1]}")
        else:
            print("Update failed - no rows returned")
            
    except Exception as e:
        print(f"Update failed with error: {e}")


def check_query_analytics_125():
    """Check specific query analytics ID 125 and its retrieval records."""
    storage = PostgresStorage()
    
    print("=== Query Analytics ID 125 ===\n")
    
    # Get query analytics record
    query = """
    SELECT id, session_id, user_id, query, timestamp, num_sources_found
    FROM query_analytics 
    WHERE id = 125;
    """
    
    result = storage._execute_query(query, fetch='one')
    
    if not result:
        print("Query Analytics ID 125 not found.")
        return
    
    qa_id, session_id, user_id, query_text, timestamp, num_sources = result
    print(f"Query Analytics ID 125:")
    print(f"  Session: {session_id}")
    print(f"  User ID: {user_id}")
    print(f"  Query: '{query_text}'")
    print(f"  Timestamp: {timestamp}")
    print(f"  Sources Found: {num_sources}")
    
    # Get retrieval analytics records
    retrieval_query = """
    SELECT id, retrieved_url, rank_position, similarity_score, is_relevant, timestamp
    FROM retrieval_analytics 
    WHERE query_analytics_id = 125
    ORDER BY rank_position;
    """
    
    retrieval_results = storage._execute_query(retrieval_query, fetch='all')
    
    print(f"\nRetrieval Analytics for QA ID 125 ({len(retrieval_results)} records):")
    for row in retrieval_results:
        ra_id, url, rank, sim_score, is_relevant, timestamp = row
        print(f"  RA ID {ra_id}: Rank {rank}")
        print(f"    URL: {url}")
        print(f"    Similarity: {sim_score}")
        print(f"    Relevant: {is_relevant}")
        print(f"    Timestamp: {timestamp}")
        print()


if __name__ == "__main__":
    try:
        check_recent_retrieval_analytics()
        check_query_analytics_125()
        test_update_query()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 