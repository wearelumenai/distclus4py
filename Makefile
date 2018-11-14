all: bindgo

bindgo:
	go build -buildmode=c-shared -x -o distclus/lib/distclus.so distclus4py/facade/
	cp facade/bind.h distclus/lib/

test: bindgo
	pipenv install
	pipenv run python setup.py test
