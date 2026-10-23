"""Generation providers (images now, video later), with a spend ledger and hard caps.

Rules learned the expensive way (credits burned on prompts nobody read):
  - every paid call goes through `ledger.spend()`, which checks the project's cap BEFORE the request
  - every call is logged with its prompt, parameters, estimate and output files
  - keys come only from the environment or a .env file (never printed)
"""
