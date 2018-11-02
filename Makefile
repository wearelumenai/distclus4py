all: bindgo

bindgo:
	go build -buildmode=c-shared -x -o distclus/lib/distclus.so distclus4py/facade/
	cp facade/bind.h distclus/lib/
