.PHONY: check reformat test

check: checkformat test

checkformat:
	python -m black --check . || (echo "Please run 'make reformat' to fix formatting issues" && exit 1)

reformat:
	python -m black .

test:
	python -m pytest   # unit tests

retest:
	python -m pytest  -x --lf  # unit tests
