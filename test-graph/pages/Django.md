tags:: #framework, #web
type:: framework
language:: [[Python]]
created:: [[2024-01-16]]

- Django is a high-level [[Python]] web framework
  - Follows the MTV (Model-Template-View) pattern
  - Created in 2005 for journalism websites
- Core components
  id:: 65a2c1f0-5678-4567-89ab-cdef01234568
  - ORM (Object-Relational Mapping)
    - Database abstraction layer
    - Supports multiple databases
  - Admin interface
    - Auto-generated admin panel
    - Highly customizable
  - Template engine
    - Django Template Language (DTL)
- Security features
  - CSRF protection built-in
  - SQL injection prevention
  - XSS protection
  - Clickjacking protection
- Django REST Framework
  - Extension for building APIs
  - Serialization and validation
  - Authentication and permissions
- Deployment considerations
  - Use gunicorn or uWSGI
  - Configure static file serving
  - Set DEBUG=False in production
    priority:: high
