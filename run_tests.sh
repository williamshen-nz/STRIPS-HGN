# Coverage results are written to the htmlcov/ directory
# Add additional modules to be covered by coverage as --cov <module name>

pytest --cov-config=.coveragerc --cov-report term --cov-report html \
  --cov=strips_hgn \
  --ignore=src/fast_downward
