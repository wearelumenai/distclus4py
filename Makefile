all: build test

GOSRC:=$(wildcard facade/*.go)

.PHONY: configci
configci:
	cd ${HOME}; \
	wget https://dl.google.com/go/go1.11.4.linux-amd64.tar.gz; \
	tar -xf go1.11.4.linux-amd64.tar.gz

.PHONY: distclus/lib/distclus.so
distclus/lib/distclus.so: ${GOSRC}
	go get -v ./facade
	go build -buildmode=c-shared -x -o distclus/lib/distclus.so github.com/wearelumenai/distclus4py/facade
	cp facade/bind.h distclus/lib/

.PHONY: build
build: gobuild pybuild

.PHONY: gobuild
gobuild: distclus/lib/distclus.so

.PHONY: pybuild
pybuild: gobuild
	python3 setup.py build
	python3 setup.py install

.PHONY: test
test: gotest pytest

.PHONY: gotest
gotest:
	go test -coverprofile=coverage.out -timeout=60000ms -short -v ./...

.PHONY: pytest
pytest: pybuild
	python setup.py test
