#!/usr/bin/env python3
"""
Script to create sample analytics data for testing the dashboard.
This creates sample user feedback, query analytics, and retrieval analytics.
"""

import sys
import os
import random
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from hubert.db.postgres_storage import PostgresStorage
import uuid

def create_sample_data():
    """Create sample analytics data for testing."""
    storage = PostgresStorage()
    
    # Sample queries and answers
    sample_interactions = [
        {
            "query": "What are the admission requirements for Computer Science?",
            "answer": "The admission requirements for Computer Science include a bachelor's degree, mathematics prerequisites, and programming experience.",
            "rating": "positive"
        },
        {
            "query": "When does the winter semester start?",
            "answer": "The winter semester typically starts in October.",
            "rating": "positive"
        },
        {
            "query": "How can I apply for a scholarship?",
            "answer": "You can apply for scholarships through the student portal. The deadline is usually in March.",
            "rating": "positive"
        },
        {
            "query": "What is the library opening hours?",
            "answer": "The library is open Monday to Friday from 8 AM to 10 PM.",
            "rating": "negative"
        },
        {
            "query": "How to register for courses?",
            "answer": "Course registration is done through the online portal during the registration period.",
            "rating": "positive"
        },
        {
            "query": "What are the graduation requirements?",
            "answer": "Graduation requirements include completing all required courses and maintaining a minimum GPA.",
            "rating": "positive"
        },
        {
            "query": "Where is the computer lab?",
            "answer": "The computer lab is located in building 123, room 456.",
            "rating": "negative"
        },
        {
            "query": "How to access online resources?",
            "answer": "Online resources can be accessed through the university portal with your student credentials.",
            "rating": "positive"
        }
    ]
    
    # Sample URLs for retrieval analytics
    sample_urls = [
        "https://www.hu-berlin.de/en/studying/admission/",
        "https://www.hu-berlin.de/en/studying/courses/",
        "https://www.hu-berlin.de/en/studying/semester/",
        "https://www.hu-berlin.de/en/studying/scholarships/",
        "https://www.hu-berlin.de/en/library/",
        "https://www.hu-berlin.de/en/studying/registration/",
        "https://www.hu-berlin.de/en/studying/graduation/",
        "https://www.hu-berlin.de/en/facilities/computer-lab/"
    ]
    
    print("Creating sample analytics data...")
    
    # Create data for the last 30 days
    for days_ago in range(30):
        date = datetime.now() - timedelta(days=days_ago)
        
        # Create 1-5 interactions per day
        num_interactions = random.randint(1, 5)
        
        for _ in range(num_interactions):
            interaction = random.choice(sample_interactions)
            session_id = str(uuid.uuid4())
            
            # Create query analytics
            query_data = {
                'session_id': session_id,
                'user_id': None,  # Anonymous users
                'query': interaction['query'],
                'has_answer': True,
                'response_time_ms': random.randint(500, 3000),
                'retrieval_method': random.choice(['vector', 'hybrid', 'keyword']),
                'num_sources_found': random.randint(3, 8)
            }
            
            query_analytics_id = storage.store_query_analytics(query_data)
            
            # Create retrieval analytics
            num_sources = query_data['num_sources_found']
            retrieved_urls = random.sample(sample_urls, min(num_sources, len(sample_urls)))
            
            retrieval_results = [
                {
                    'url': url,
                    'rank_position': idx + 1,
                    'similarity_score': random.uniform(0.7, 0.95)
                }
                for idx, url in enumerate(retrieved_urls)
            ]
            
            storage.store_retrieval_analytics(query_analytics_id, retrieval_results)
            
            # Create user feedback (not for all queries)
            if random.random() < 0.6:  # 60% chance of feedback
                feedback_data = {
                    'session_id': session_id,
                    'user_id': None,
                    'query': interaction['query'],
                    'generated_answer': interaction['answer'],
                    'prompt_used': random.choice(['default', 'academic', 'concise']),
                    'retrieval_method': query_data['retrieval_method'],
                    'sources_urls': retrieved_urls[:3],  # Top 3 sources
                    'rating': interaction['rating'],
                    'feedback_comment': None,
                    'response_time_ms': query_data['response_time_ms']
                }
                
                storage.store_user_feedback(feedback_data)
            
            print(f"Created interaction for {date.strftime('%Y-%m-%d')}: {interaction['query'][:50]}...")
    
    print(f"\n✅ Sample data creation completed!")
    print("You can now test the analytics dashboard at /analytics")

if __name__ == "__main__":
    create_sample_data() 