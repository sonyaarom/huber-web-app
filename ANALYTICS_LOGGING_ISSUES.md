# Analytics Logging Issues and Solutions

## Issues Identified

### 1. Multiple Query Analytics IDs for One Search Request

**Problem**: Your search for "stefan" created 4 different query analytics records (IDs 121, 122, 123, 124) instead of one.

**Root Cause**: Each time you clicked thumbs up/down on a search result, the `submit_search_source_feedback()` function was creating a **new** query analytics record instead of reusing the existing one.

**Evidence from your data**:
```csv
query_analytics.csv:
"121","search-session-1750805941292","6","stefan" - Original search (15812ms, 5 sources)
"122","search-session-1750890639894","6","stefan" - Feedback #1 (0ms, 1 source)  
"123","search-session-1750890640773","6","stefan" - Feedback #2 (0ms, 1 source)
"124","search-session-1750890641522","6","stefan" - Feedback #3 (0ms, 1 source)
```

**Fix Applied**: Modified `ui/app/views/main.py` to:
- First check for existing query analytics record by session + query
- Fall back to user + query lookup within 2 hours
- Only create new record if none exists
- Added better logging to track what's happening

### 2. Missing Similarity Scores

**Problem**: Only 1 out of 4 retrieval analytics records has similarity scores.

**Root Cause**: Similarity scores are only captured during the initial search retrieval. When feedback creates new retrieval analytics records, they use `similarity_score: None`.

**Evidence**:
```csv
retrieval_analytics.csv:
"640","121",...,"2.34446382522583" - Has similarity score (original search)
"642","122",...,"" - Missing similarity score (feedback-generated)
"643","123",...,"" - Missing similarity score (feedback-generated)  
"644","124",...,"" - Missing similarity score (feedback-generated)
```

**Fix Applied**: The code now reuses existing query analytics records, so similarity scores are preserved from the original search.

### 3. Session ID Management Issues

**Problem**: Different session IDs being generated for feedback on the same search.

**Root Cause**: Frontend search page generates random session IDs like `search-session-1750890641522` for each feedback action.

**Fix Applied**: 
- Better session ID persistence in the feedback handling
- Lookup by user ID as fallback when session matching fails

### 4. Timezone Issues

**Problem**: Timestamps show UTC timezone (`+00:00`) which might not match your local time.

**Analysis**: This is actually **correct behavior**:
- Database properly stores timestamps in UTC  
- All timestamps are timezone-aware (`DateTime(timezone=True)`)
- PostgreSQL `now()` function returns UTC time

**Recommendation**: Add timezone conversion in the UI for better user experience, but keep database storage in UTC.

## How to Identify Records from One Search Request

With the fixes applied, you can identify records from one search using:

### Method 1: Query Analytics ID (Recommended)
```sql
-- Get all retrieval results for a specific search
SELECT ra.*, qa.query, qa.session_id 
FROM retrieval_analytics ra
JOIN query_analytics qa ON ra.query_analytics_id = qa.id  
WHERE qa.id = 121;  -- Use the query_analytics_id
```

### Method 2: Session ID + Query
```sql
-- Get all records for a specific search session
SELECT qa.id, qa.query, qa.timestamp, ra.retrieved_url, ra.rank_position
FROM query_analytics qa
LEFT JOIN retrieval_analytics ra ON qa.id = ra.query_analytics_id
WHERE qa.session_id = 'search-session-1750805941292' 
AND qa.query = 'stefan';
```

### Method 3: User + Query + Time Window  
```sql
-- Get recent searches by user for a specific query
SELECT qa.id, qa.query, qa.timestamp, COUNT(ra.id) as retrieval_count
FROM query_analytics qa  
LEFT JOIN retrieval_analytics ra ON qa.id = ra.query_analytics_id
WHERE qa.user_id = 6 
AND qa.query = 'stefan'
AND qa.timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY qa.id, qa.query, qa.timestamp
ORDER BY qa.timestamp DESC;
```

## Database Schema Relationships

```
query_analytics (Main record for each search)
├── id (Primary key)
├── session_id (Session identifier)  
├── user_id (User who searched)
├── query (The search text)
├── timestamp (When search happened)
└── ... (other metadata)

retrieval_analytics (Results for each search)
├── id (Primary key)
├── query_analytics_id (Foreign key → query_analytics.id)  
├── retrieved_url (The result URL)
├── rank_position (1, 2, 3, etc.)
├── similarity_score (Relevance score)
├── is_relevant (User feedback: true/false/null)
└── timestamp (When result was retrieved/feedback given)
```

## Running the Analysis Script

To analyze your current data and identify issues:

```bash
cd /Users/sonyarom/Entwicklung/HuBer
python scripts/analyze_analytics_data.py
```

This will show:
- Duplicate query analytics records  
- Missing similarity scores
- Suggestions for cleanup
- SQL queries to fix existing data

## Testing the Fixes

1. **Clear browser session storage** to reset session IDs
2. **Perform a new search** 
3. **Submit feedback** on multiple results
4. **Check the analytics tables** - should see:
   - Only 1 query_analytics record for the search
   - Multiple retrieval_analytics records with same query_analytics_id
   - Similarity scores preserved from original search
   - Consistent session_id across all records

## Next Steps

1. **Deploy the code fixes**
2. **Run the analysis script** to understand current data state  
3. **Clean up existing duplicates** (optional - use the provided SQL)
4. **Consider adding timezone display conversion** in the UI
5. **Monitor new searches** to confirm fixes work correctly

The key insight is that `query_analytics_id` is your primary identifier for grouping all records related to one search request. Each search should have exactly one query_analytics record and multiple retrieval_analytics records linked to it. 