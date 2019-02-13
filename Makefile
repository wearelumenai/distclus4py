all: build test

configci:
	cd ${HOME}; \
	wget https://dl.google.com/go/go1.11.4.linux-amd64.tar.gz; \
	tar -xf go1.11.4.linux-amd64.tar.gz
	cd ${GOPATH}/src ; \
	git clone git@github.com:/wearelumenai/distclus.git

build:
	go get -v ./facade
	go build -buildmode=c-shared -x -o distclus/lib/distclus.so distclus4py/facade/
	cp facade/bind.h distclus/lib/

test:
	go test -coverprofile=coverage.out -timeout=60000ms -short -v ./...
	pipenv install "-e .[test]"
	pipenv run py.test --cov=distclus tests
