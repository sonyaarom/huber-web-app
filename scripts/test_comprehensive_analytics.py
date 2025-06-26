#!/usr/bin/env python3
"""
Test script for comprehensive analytics functionality.
"""
import sys
import os
import json
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hubert.db.postgres_storage import PostgresStorage


def test_comprehensive_analytics():
    """Test the comprehensive analytics functionality."""
    storage = PostgresStorage()
    
    print("=== Testing Comprehensive Analytics ===\n")
    
    try:
        # Test 7-day metrics
        print("Testing 7-day metrics...")
        metrics_7d = storage.get_comprehensive_analytics_metrics(days=7)
        
        print("7-day metrics structure:")
        for key, value in metrics_7d.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} records")
                if value and len(value) > 0:
                    print(f"    Sample: {value[0]}")
            elif isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            else:
                print(f"  {key}: {value}")
        
        print("\n" + "="*50 + "\n")
        
        # Test 30-day metrics
        print("Testing 30-day metrics...")
        metrics_30d = storage.get_comprehensive_analytics_metrics(days=30)
        
        print("30-day metrics structure:")
        for key, value in metrics_30d.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} records")
            elif isinstance(value, dict):
                print(f"  {key}: {list(value.keys())}")
            else:
                print(f"  {key}: {value}")
        
        print("\n" + "="*50 + "\n")
        
        # Display detailed summary stats
        print("📊 Summary Statistics (7 days):")
        stats = metrics_7d.get('summary_stats', {})
        print(f"  🔍 Total Requests: {stats.get('total_requests', 0)}")
        print(f"  💬 Unique Queries: {stats.get('total_unique_queries', 0)}")
        print(f"  ⚡ Avg Response Time: {stats.get('avg_response_time', 0):.2f}ms")
        print(f"  💬 Total Feedback: {stats.get('total_feedback_given', 0)}")
        print(f"  👍 Positive: {stats.get('positive_feedback', 0)}")
        print(f"  👎 Negative: {stats.get('negative_feedback', 0)}")
        
        feedback_rate = (stats.get('total_feedback_given', 0) / stats.get('total_requests', 1) * 100) if stats.get('total_requests', 0) > 0 else 0
        print(f"  📝 Feedback Rate: {feedback_rate:.1f}%")
        
        print(f"\n🎯 MRR: {metrics_7d.get('mrr_metrics', {}).get('overall_mrr', 0):.3f}")
        
        print("\n🔥 Most Popular Queries:")
        for i, query in enumerate(metrics_7d.get('popular_queries', [])[:5], 1):
            print(f"  {i}. '{query[0]}' - {query[1]} times, {query[2]:.0f}ms avg")
        
        print("\n🏆 Top URLs (by relevance):")
        for i, url in enumerate(metrics_7d.get('top_urls', [])[:5], 1):
            precision = (url[2] / url[1]) if url[1] > 0 else 0
            print(f"  {i}. {url[0].split('/')[-1] or url[0].split('/')[-2]} - {url[2]}/{url[1]} relevant ({precision:.1%})")
        
        print("\n☁️ Popular Terms:")
        terms = [f"{term[0]}({term[1]})" for term in metrics_7d.get('popular_terms', [])[:10]]
        print(f"  {', '.join(terms)}")
        
        print("\n✅ Comprehensive analytics test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing comprehensive analytics: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_sample_data():
    """Export sample data for debugging."""
    storage = PostgresStorage()
    
    try:
        metrics = storage.get_comprehensive_analytics_metrics(days=7)
        
        # Save to file for inspection
        output_file = "sample_analytics_output.json"
        with open(output_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            json.dump(metrics, f, indent=2, default=convert_datetime)
        
        print(f"✅ Sample analytics data exported to {output_file}")
        
    except Exception as e:
        print(f"❌ Error exporting sample data: {e}")


if __name__ == "__main__":
    if test_comprehensive_analytics():
        export_sample_data()
    else:
        print("❌ Analytics test failed - not exporting data") 