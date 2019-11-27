all: build test

GOSRC:=$(wildcard facade/*.go)

configci:
	cd ${HOME}; \
	wget https://dl.google.com/go/go1.11.4.linux-amd64.tar.gz; \
	tar -xf go1.11.4.linux-amd64.tar.gz
	cd ${GOPATH}/src ; \
	git clone git@github.com:/wearelumenai/distclus.git

distclus/lib/distclus.so: ${GOSRC}
	go get -v ./facade
	go build -buildmode=c-shared -x -o distclus/lib/distclus.so distclus4py/facade/
	cp facade/bind.h distclus/lib/

build: distclus/lib/distclus.so

test: gotest pytest

gotest:
	go test -coverprofile=coverage.out -timeout=60000ms -short -v ./...

pybuild:
	pipenv install "-e .[test]"

pytest: pybuild
	pipenv run py.test --cov=distclus tests
