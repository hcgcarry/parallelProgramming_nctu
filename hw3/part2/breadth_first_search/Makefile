all: default grade

default: main.cpp bfs.cpp
	g++ -I../ -std=c++17 -fopenmp  -g -o bfs main.cpp bfs.cpp ../common/graph.cpp ref_bfs.a
grade: grade.cpp bfs.cpp
	g++ -I../ -std=c++17 -fopenmp  -g -o bfs_grader grade.cpp bfs.cpp ../common/graph.cpp ref_bfs.a
clean:
	rm -rf bfs_grader bfs  *~ *.*~

test:
	./bfs ../graph/all_graphs/grid10x10.graph
time:
	time ./bfs ../graph/all_graphs/grid1000x1000.graph

	
perf:
	perf record ./bfs ../graph/all_graphs/grid1000x1000.graph