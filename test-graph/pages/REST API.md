tags:: #api, #architecture
type:: pattern
created:: [[2024-01-27]]

- REST (Representational State Transfer) is an architectural style
  - Created by Roy Fielding in 2000
  - Stateless client-server communication
  - Uses HTTP methods and status codes
- HTTP methods
  id:: 65a2c1f0-8888-4567-89ab-cdef01234572
  - GET: retrieve resources
    - Should be idempotent and safe
  - POST: create new resources
  - PUT: update/replace resources
  - PATCH: partial updates
  - DELETE: remove resources
- Key principles
  - Resource-based (not action-based)
  - Stateless operations
  - Cacheable responses
  - Uniform interface
  - Layered system
- URL design
  - Use nouns, not verbs
    - /users not /getUsers
  - Plural for collections
  - Nested resources show relationships
    - /users/123/posts
  - Query parameters for filtering
    - /users?role=admin
- Status codes
  - 200 OK: successful GET
  - 201 Created: successful POST
  - 204 No Content: successful DELETE
  - 400 Bad Request: client error
  - 401 Unauthorized: authentication required
  - 404 Not Found: resource doesn't exist
  - 500 Internal Server Error: server error
- Best practices
  - Version your API (/v1/users)
  - Use JSON for request/response
  - Implement pagination
  - Provide filtering and sorting
  - Include proper error messages
  - Document with OpenAPI/Swagger
- Alternatives
  - GraphQL for flexible queries
  - gRPC for high performance
  - WebSocket for real-time
