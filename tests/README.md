# Sitemap Parsing Tests

This directory contains comprehensive tests for the sitemap parsing functionality in HUBer. The tests verify that sitemap changes are correctly detected and processed.

## 📋 Test Coverage

### Unit Tests (`test_sitemap_parsing.py`)
- **XML Parsing**: Validates standard XML sitemap parsing
- **Regex Fallback**: Tests fallback to regex when XML parsing fails
- **Filtering**: Verifies URL filtering by extensions, patterns, and base URL
- **Security**: Tests security validation of URLs
- **Error Handling**: Covers network errors and malformed content

### Scenario Tests (`test_sitemap_changes_scenario.py`)
- **New Pages**: Tests detection and processing of new URLs
- **Removed Pages**: Verifies that removed URLs are marked as inactive
- **Updated Pages**: Checks that timestamp updates are handled correctly
- **Mixed Changes**: Tests complex scenarios with multiple change types
- **Format Support**: Validates gzipped and various sitemap formats
- **Filtering**: Tests real-world filtering scenarios

## 🚀 Running the Tests

### Quick Start
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
python run_sitemap_tests.py

# Or use pytest directly
pytest tests/ -v
```

### Specific Test Types
```bash
# Run only unit tests
python run_sitemap_tests.py --type unit

# Run only scenario tests
python run_sitemap_tests.py --type scenarios

# Run only integration tests (requires database)
python run_sitemap_tests.py --type integration

# Run with coverage
pytest tests/ --cov=hubert.data_ingestion.huber_crawler --cov-report=html
```

## 🧪 Test Scenarios

### Scenario 1: New Pages Added
```python
def test_new_pages_added_to_sitemap():
    """
    Tests what happens when new pages are added to the sitemap.
    
    Expected behavior:
    - New URLs are added to the database
    - Existing URLs remain unchanged
    - Database operations are called correctly
    """
```

### Scenario 2: Pages Removed
```python
def test_pages_removed_from_sitemap():
    """
    Tests what happens when pages are removed from the sitemap.
    
    Expected behavior:
    - Removed URLs are marked as inactive
    - Active URLs remain in the database
    - Cleanup operations are performed
    """
```

### Scenario 3: Pages Updated
```python
def test_pages_updated_in_sitemap():
    """
    Tests what happens when existing pages have updated timestamps.
    
    Expected behavior:
    - URLs are updated with new lastmod dates
    - Content processing is triggered for updated pages
    - Database reflects the new timestamps
    """
```

### Scenario 4: Mixed Changes
```python
def test_mixed_changes_in_sitemap():
    """
    Tests complex scenarios with multiple types of changes.
    
    Expected behavior:
    - New pages are added
    - Removed pages are deactivated
    - Updated pages have new timestamps
    - All operations are performed in correct order
    """
```

## 📊 Sample Test Data

The tests use realistic sitemap data modeled after your actual website:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://www.wiwi.hu-berlin.de/en/research</loc>
        <lastmod>2024-01-15T10:30:00Z</lastmod>
    </url>
    <url>
        <loc>https://www.wiwi.hu-berlin.de/en/teaching</loc>
        <lastmod>2024-01-16T11:00:00Z</lastmod>
    </url>
</urlset>
```

## 🔧 Mock Components

### PostgresStorage Mock
The tests use a comprehensive mock of the `PostgresStorage` class:
- `upsert_raw_pages()` - Tracks URL insertions/updates
- `deactivate_old_urls()` - Tracks URL deactivations
- `connect()` and `close()` - Database connection management

### HTTP Request Mocking
Tests can mock HTTP requests to simulate:
- Different sitemap content
- Network errors
- Gzipped responses
- Various HTTP status codes

## 🎯 Test Assertions

### Database Operations
```python
# Verify that new URLs are inserted
mock_storage.upsert_raw_pages.assert_called_once()
upserted_urls = {record['url'] for record in mock_storage.upsert_raw_pages.call_args[0][0]}
assert 'https://www.wiwi.hu-berlin.de/en/new-page' in upserted_urls

# Verify that old URLs are deactivated
mock_storage.deactivate_old_urls.assert_called_once()
```

### Content Validation
```python
# Verify parsing results
assert len(records) == expected_count
assert all('url' in data and 'last_updated' in data for data in records.values())

# Verify URL filtering
filtered_urls = {data['url'] for data in records.values()}
assert filtered_urls == expected_urls
```

### Metrics Tracking
```python
# Verify metrics are updated
assert metrics.total_urls_found == expected_count
assert metrics.errors == 0
assert metrics.database_update_time > 0
```

## 🐛 Common Issues

### Import Errors
If you see import errors, ensure the project root is in your Python path:
```python
import sys
sys.path.insert(0, '/path/to/HUBer')
```

### Missing Dependencies
Install all test dependencies:
```bash
pip install -r requirements-test.txt
```

### Environment Variables
Some tests may require database connection details in your `.venv` file.

## 📈 Continuous Integration

To run these tests in CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run Sitemap Tests
  run: |
    pip install -r requirements-test.txt
    pytest tests/ --cov=hubert.data_ingestion.huber_crawler --cov-report=xml
```

## 🔍 Debugging Tests

### Verbose Output
```bash
pytest tests/ -v -s  # Show print statements
```

### Single Test
```bash
pytest tests/test_sitemap_changes_scenario.py::TestSitemapChangesScenarios::test_new_pages_added_to_sitemap -v
```

### Debug Mode
```bash
pytest tests/ --pdb  # Drop into debugger on failure
```

## 📝 Adding New Tests

When adding new test scenarios:

1. **Create Realistic Data**: Use actual URL patterns from your website
2. **Test Edge Cases**: Consider malformed XML, network issues, empty sitemaps
3. **Verify All Operations**: Check both database operations and metrics
4. **Use Fixtures**: Leverage the existing fixtures in `conftest.py`
5. **Document Expected Behavior**: Add clear docstrings explaining what should happen

Example:
```python
def test_your_new_scenario(self):
    """
    Test description: What scenario you're testing
    
    Expected behavior:
    - What should happen in the database
    - What metrics should be updated
    - What operations should be called
    """
    # Test implementation
```

## 🎉 Success Criteria

Tests pass when:
- ✅ All URL changes are correctly detected
- ✅ Database operations are called with correct parameters
- ✅ Filtering works as expected
- ✅ Error conditions are handled gracefully
- ✅ Metrics accurately reflect processing results
- ✅ No unexpected exceptions are raised 