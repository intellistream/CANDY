
# chekc cmake exists
ifeq ($(shell which cmake),)
$(error "cmake is not installed")
endif

threads=$(shell nproc)

all: build

build:
	mkdir -p build
	cd build && cmake .. && cmake --build . -- -j$(threads)
clean:
	rm -rf build