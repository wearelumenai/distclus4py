all: bindgo test

bindgo:
	go build -buildmode=c-shared -x -o distclus/lib/distclus.so distclus4py/facade/
	cp facade/bind.h distclus/lib/

test:
	go test ./...
	pipenv install "-e .[test]"
	pipenv run python setup.py test
	pipenv run py.test --cov=distclus tests
