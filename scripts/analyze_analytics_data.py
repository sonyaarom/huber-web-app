#!/usr/bin/env python3
"""
Script to analyze analytics data and identify issues with logging.
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hubert.db.postgres_storage import PostgresStorage


def analyze_query_analytics():
    """Analyze query analytics data for issues."""
    storage = PostgresStorage()
    
    print("=== Query Analytics Analysis ===\n")
    
    # Get all query analytics data
    query = """
    SELECT id, session_id, user_id, query, timestamp, retrieval_method, num_sources_found
    FROM query_analytics 
    ORDER BY timestamp DESC
    LIMIT 50
    """
    
    results = storage._execute_query(query, fetch='all')
    
    if not results:
        print("No query analytics data found.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results, columns=[
        'id', 'session_id', 'user_id', 'query', 'timestamp', 'retrieval_method', 'num_sources_found'
    ])
    
    print(f"Total query analytics records: {len(df)}")
    print(f"Unique queries: {df['query'].nunique()}")
    print(f"Unique sessions: {df['session_id'].nunique()}")
    
    # Check for duplicate queries
    duplicate_queries = df.groupby(['query', 'user_id']).size().reset_index(name='count')
    duplicate_queries = duplicate_queries[duplicate_queries['count'] > 1]
    
    if not duplicate_queries.empty:
        print(f"\n⚠️  Found {len(duplicate_queries)} queries with duplicates:")
        for _, row in duplicate_queries.iterrows():
            query_text = row['query']
            user_id = row['user_id']
            count = row['count']
            
            # Get details for this duplicate
            details = df[(df['query'] == query_text) & (df['user_id'] == user_id)]
            print(f"\nQuery: '{query_text}' (User ID: {user_id}) - {count} records:")
            for _, detail in details.iterrows():
                print(f"  ID: {detail['id']}, Session: {detail['session_id'][:20]}..., Time: {detail['timestamp']}")
    
    return df


def analyze_retrieval_analytics():
    """Analyze retrieval analytics data for issues."""
    storage = PostgresStorage()
    
    print("\n=== Retrieval Analytics Analysis ===\n")
    
    # Get retrieval analytics with query info
    query = """
    SELECT ra.id, ra.query_analytics_id, qa.query, qa.session_id, ra.retrieved_url, 
           ra.rank_position, ra.similarity_score, ra.is_relevant, ra.timestamp
    FROM retrieval_analytics ra
    JOIN query_analytics qa ON ra.query_analytics_id = qa.id
    ORDER BY ra.timestamp DESC
    LIMIT 50
    """
    
    results = storage._execute_query(query, fetch='all')
    
    if not results:
        print("No retrieval analytics data found.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=[
        'id', 'query_analytics_id', 'query', 'session_id', 'retrieved_url', 
        'rank_position', 'similarity_score', 'is_relevant', 'timestamp'
    ])
    
    print(f"Total retrieval analytics records: {len(df)}")
    
    # Check similarity scores
    missing_scores = df[df['similarity_score'].isnull()]
    print(f"Records missing similarity scores: {len(missing_scores)} ({len(missing_scores)/len(df)*100:.1f}%)")
    
    # Group by query analytics ID to see structure
    grouped = df.groupby('query_analytics_id').agg({
        'query': 'first',
        'session_id': 'first',
        'id': 'count',
        'similarity_score': lambda x: x.notna().sum(),
        'is_relevant': lambda x: x.notna().sum()
    }).rename(columns={'id': 'retrieval_count', 'similarity_score': 'has_similarity_count', 'is_relevant': 'has_relevance_count'})
    
    print(f"\nQuery Analytics breakdown:")
    print(f"Average retrieval records per query: {grouped['retrieval_count'].mean():.1f}")
    
    # Show examples of issues
    print(f"\nExamples of queries with missing similarity scores:")
    problematic = grouped[grouped['has_similarity_count'] == 0].head(5)
    for qa_id, row in problematic.iterrows():
        print(f"  Query Analytics ID {qa_id}: '{row['query']}' - {row['retrieval_count']} retrievals, no similarity scores")
    
    return df


def suggest_fixes():
    """Suggest fixes for the identified issues."""
    print("\n=== Suggested Fixes ===\n")
    
    print("1. **Duplicate Query Analytics Issue:**")
    print("   - The issue is in submit_search_source_feedback() creating new query records")
    print("   - Fixed in the code changes above")
    print("   - Consider cleaning up existing duplicates")
    
    print("\n2. **Missing Similarity Scores:**")
    print("   - Similarity scores are only captured during initial search")
    print("   - Feedback-generated retrieval records don't have similarity scores")
    print("   - Consider storing similarity scores during initial search for all results")
    
    print("\n3. **Session Management:**")
    print("   - Search page generates new session IDs for each feedback")
    print("   - Should reuse session ID from the original search")
    print("   - Improved in the code changes above")
    
    print("\n4. **Timezone Display:**")
    print("   - Timestamps are correctly stored in UTC")
    print("   - Consider adding timezone conversion in the UI for better user experience")


def cleanup_duplicate_queries():
    """Provide a cleanup script for duplicate queries."""
    print("\n=== Cleanup Script ===\n")
    
    cleanup_sql = """
    -- Find and merge duplicate query analytics
    WITH duplicates AS (
        SELECT query, user_id, session_id, MIN(id) as keep_id, array_agg(id) as all_ids
        FROM query_analytics 
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY query, user_id, session_id
        HAVING COUNT(*) > 1
    )
    SELECT 
        'UPDATE retrieval_analytics SET query_analytics_id = ' || keep_id || 
        ' WHERE query_analytics_id IN (' || array_to_string(all_ids[2:], ',') || ');' as update_cmd,
        'DELETE FROM query_analytics WHERE id IN (' || array_to_string(all_ids[2:], ',') || ');' as delete_cmd
    FROM duplicates;
    """
    
    print("To clean up duplicates, you can run this SQL (BACKUP FIRST!):")
    print(cleanup_sql)


if __name__ == "__main__":
    try:
        query_df = analyze_query_analytics()
        retrieval_df = analyze_retrieval_analytics()
        suggest_fixes()
        cleanup_duplicate_queries()
        
    except Exception as e:
        print(f"Error analyzing data: {e}")
        import traceback
        traceback.print_exc() 