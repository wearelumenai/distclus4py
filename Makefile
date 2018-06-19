all: bindgo

bindgo:
	go build -buildmode=c-shared -x -o bind/build/distclus.so bind/main.go
