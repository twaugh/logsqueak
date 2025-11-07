tags:: #tools, #version-control
type:: tool
created:: [[2024-01-22]]

- Git is a distributed version control system
  Unlike centralized version control systems, Git allows every developer to have a complete copy of the entire repository history.
  - Created by Linus Torvalds in 2005
    Developed to manage Linux kernel development after the previous VCS became unavailable.
  - Industry standard for source control
- Basic concepts
  id:: 65a2c1f0-3333-4567-89ab-cdef0123456d
  - Repository (repo)
    - Contains project history
  - Commit
    - Snapshot of changes
    - Has unique SHA-1 hash
  - Branch
    - Independent line of development
  - Remote
    - Version of repo hosted elsewhere
- Common commands
  - git clone - copy a repository
  - git add - stage changes
  - git commit - save changes
  - git push - send to remote
  - git pull - fetch and merge
  - git merge - combine branches
    - Can cause merge conflicts
- Branching strategies
  Different teams adopt different branching models based on their deployment frequency and team size.
  - Git Flow
    A more complex model with multiple long-lived branches for different purposes.
    - Feature branches, develop, main
  - GitHub Flow
    - Simple: main + feature branches
  - Trunk-based development
    Favored by high-performing teams for continuous delivery.
    - Short-lived branches
- Best practices
  - Write descriptive commit messages
    - Follow conventional commits
  - Commit frequently
  - Keep commits atomic
  - Review before pushing
- Advanced features
  - Interactive rebase for history cleanup
  - Cherry-pick for selective commits
  - Bisect for bug hunting
  - Stash for temporary storage
