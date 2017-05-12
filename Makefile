
CXXFLAGS=-g -Wall -Wextra -O2 -std=c++11
CXX=c++
LDFLAGS=`pkg-config --libs opencv`

main: main.o
	${CXX} $^ -o $@ ${LDFLAGS}

main.o: main.cpp

clean:
	rm -f *.o *~ main