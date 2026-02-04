br: build
	./test
	
build:
	gcc -O3  test.c \
    -I./raylib/include \
    ./lib/libraylib.a \
    -framework Cocoa \
    -framework OpenGL \
    -framework IOKit \
    -framework CoreVideo \
    -framework CoreFoundation \
    -o test
