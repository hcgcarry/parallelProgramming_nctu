all: default grade

default: page_rank.cpp main.cpp
	g++ -I../ -std=c++17 -fopenmp -O3 -o pr main.cpp page_rank.cpp ../common/graph.cpp ref_pr.a -g 
grade: page_rank.cpp grade.cpp
	g++ -I../ -std=c++17 -fopenmp -O3 -o pr_grader grade.cpp page_rank.cpp ../common/graph.cpp ref_pr.a
clean:
	rm -rf pr pr_grader *~ *.*~

test:
	./pr ../graph/all_graphs/tiny.graph
test1:
	./pr ../graph/all_graphs/grid4x4.graph