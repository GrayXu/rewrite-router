APP = rewrite-router
GOFILES = main.go

.PHONY: all build run tidy clean fmt

all: build

build:
	go build -o $(APP) $(GOFILES)

run:
	go run $(GOFILES)

tidy:
	go mod tidy

fmt:
	gofmt -w .

clean:
	rm -f $(APP) 