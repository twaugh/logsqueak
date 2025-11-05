tags:: #api, #query-language
type:: technology
created:: [[2024-01-28]]

- GraphQL is a query language for APIs
  - Developed by Facebook in 2012
  - Alternative to [[REST API]]
  - Client specifies exact data needed
- Key advantages
  id:: 65a2c1f0-9999-4567-89ab-cdef01234573
  - No over-fetching or under-fetching
    - Get exactly what you request
  - Single endpoint
  - Strong typing
  - Introspection
  - Real-time with subscriptions
- Core concepts
  - Schema definition
    - Types and fields
  - Queries for reading data
  - Mutations for writing data
  - Subscriptions for real-time updates
  - Resolvers implement field logic
- Schema example
  - type User { id: ID!, name: String! }
  - type Query { user(id: ID!): User }
  - type Mutation { createUser(name: String!): User }
- Advantages over [[REST API]]
  - Versionless API evolution
  - Reduced network requests
  - Better developer experience
  - Self-documenting
- Challenges
  - Complexity for simple use cases
  - Caching more difficult
  - File uploads require extensions
  - N+1 query problem
    - Use DataLoader to solve
- [[Python]] implementations
  - Graphene (most popular)
  - Strawberry (modern, uses dataclasses)
  - Ariadne (schema-first approach)
- When to choose GraphQL
  - Complex data requirements
  - Mobile applications
  - Multiple clients with different needs
  - Rapid iteration
