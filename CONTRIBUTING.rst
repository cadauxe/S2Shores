================================
Contributing to Project S2Shores
================================

This document provides guidelines and information on how to contribute effectively to the project.

Report Issues
=============

If you encounter any issues, bugs, or have feature requests, please report them through the project's issue tracker. Provide as much detail as possible to help us understand and reproduce the problem. Include:
- A clear and descriptive title
- Steps to reproduce the issue
- Expected and actual results
- Relevant screenshots or logs

Contributing Workflow
=====================

To contribute to Project S2Shores, follow these steps:

1. **Fork the repository:**
Create a personal fork of the repository on GitHub.

2. **Clone your fork:**
Clone your forked repository to your local machine:
```bash
git clone https://github.com/your-username/S2Shores.git
cd S2Shores
```

3. **Create a new branch:**
Create a new branch for your feature or bugfix:

```bash
git checkout -b feature-or-bugfix-name
```

4. **Make your changes:**
Implement your changes in the new branch. Follow the coding guide and ensure your code meets the quality standards.


5. **Run linting and tests:**
Ensure your code passes Pylint checks and all tests:
```bash
pylint .
pytest
```

6. **Commit your changes:**

Use the hooks provided to check your code quality before committing as described below.

Commit your changes with a descriptive commit message:

```bash
git add .
git commit -m "Brief description of your changes"
```

7. **Push to your fork:**
Push your branch to your forked repository:
```bash
git push origin feature-or-bugfix-name
```

8. **Create a Pull Request:**
Open a pull request (PR) from your branch to the main repository. Provide a detailed description of your changes and the issue it addresses.


Handling Issues in Your Workflow
=================================
1. Linking Commits to Issues

Creating a Branch for the Issue
--------------------------------

When you start working on a new issue, create a new branch with a descriptive name that includes the github issue number:

```bash
git checkout -b 42-fix-bug-in-module
```

This helps keep your work organized and makes it easy to track the changes related to a specific issue.

Committing with Issue References
--------------------------------

When committing changes related to an issue, include the issue number in your commit message. This practice helps in linking the commit to the issue automatically in platforms like GitHub and GitLab:

```bash
git commit -m "Fix bug in module X related to #42"
```
The commit will be automatically linked to the issue #42 in the issue tracker.

Pushing the Branch
-------------------

Push your branch to the remote repository:

```bash
git push origin 42-fix-bug-in-module
```

This allows others to review your work and ensures that the issue tracking is kept up to date.


2. Closing Issues and creating a merge request

Closing Issues
--------------

If your commit resolves an issue, you can automatically close the issue by using keywords in your commit message:

```bash
git commit -m "Fix bug in module X, closes #42"
```
This will close the issue when the commit is merged into the main branch.

Creating a Merge Request/ Pull Request
--------------------------------------

When your work is complete, create a pull request (GitHub) from your branch to the main branch.
In the description, reference the issue number again to ensure clarity:

```markdown
This merge request addresses issue #42 by fixing the bug in module X.
```

This practice ensures that your work is well-documented.


Coding Guide
============

Please adhere to the following coding guidelines to ensure consistency and quality across the codebase:

- **Follow PEP 8:** The project follows the PEP 8 style guide for Python code. Use Pylint to check your code for compliance.
- **Use meaningful names:** Choose meaningful names for variables, functions, and classes.
- **Write docstrings:** Provide clear and concise docstrings for all functions and classes.
- **Keep functions small:** Aim to keep functions short and focused on a single task.
- **Use type annotations:** Use type annotations to improve code readability and help with static analysis.

Pylint and Git Hooks
====================

To maintain code quality, we use Pylint and Git hooks. Pylint checks for errors, enforces coding standards, and helps identify code smells.

Setting up Pylint
-----------------

The project includes a `.pylintrc` file with the necessary configurations. Ensure Pylint is installed and set up to use this configuration file:

```bash
pip install pylint
```

Git Hooks
---------

Git hooks are used to automate quality checks before commits and pushes. The hooks ensure that Pylint and tests are run automatically.

Setup Hooks:
~~~~~~~~~~~~
To set up the Git hooks, run the following command in the project directory:
```bash
git config --local core.hooksPath .githooks/
```

Pre-Commit Hook:
~~~~~~~~~~~~~~~~
The pre-commit hook runs Pylint on staged files of the project before allowing a commit. If any checks fail, the commit is aborted.

Pre-Push Hook:
~~~~~~~~~~~~~~
The pre-push hook runs Pylint on the entire project and pytest before allowing a push. If any checks fail, the push is aborted.

Merge Request Acceptance Process
================================

To ensure high quality and consistency in the codebase, follow these steps for getting your merge request accepted:

Review and Approval:
--------------------
Your pull request will be reviewed by project maintainers. They may provide feedback or request changes.

Address Feedback:
-----------------
Make the necessary changes based on the feedback and update your pull request.

Automated Checks:
-----------------
Ensure all automated checks (linting, tests) pass. The pull request must pass all checks before it can be merged.

Final Review:
-------------
After addressing feedback and passing all checks, the maintainers will perform a final review. If everything is satisfactory, your pull request will be merged.
