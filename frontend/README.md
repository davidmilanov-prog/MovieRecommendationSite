### Frontend Integration
This frontend is a simple React UI that talks to the FastAPI backend.

The UI has two main inputs:

1. **Mode Dropdown**
- "Standard Search" (id 0) = semantic search only
- Archetypes (ids 1000000+) = semantic search with prompt bias
- Random User (a real MovieLens user) = hybrid semantic retrieval + CF re-rank

2. **Search Bar**
Users type natural language queries like:
- "90s thriller with a twist"
- "funny romcom"
- "space sci-fi action"

### Backend Endpoints Used
- `GET /personas` to populate the dropdown
- `GET /random_user` when clicking Random User
- `POST /recommend` to fetch recommendations
