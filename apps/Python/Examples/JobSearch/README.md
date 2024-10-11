# Real-Life Use Case: Real-Time Job Postings and Job Search

Use Case Overview:
- Job postings are continuously added to the system.
- Each job posting contains both unstructured data (job description text, which needs to be embedded) and structured
- metadata (job type, location, salary range).

- Users can query the system using a combination of:
- Text-based queries (e.g., "data science jobs")
- Structured filters (e.g., filter by location, salary range)

The goal is to use the Flink-style API to continuously ingest and transform incoming job data while allowing users to
make hybrid queries that filter results based on both embedding similarity and structured metadata.