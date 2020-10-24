


library_FLAGS=-lpthread -lm  -mavx2  -Ofast


all:pi.out

pi.out:pi
%:%.c
	gcc $< -o $@ $(library_FLAGS)
%:%.cpp
	g++ $< -o $@ $(library_FLAGS)

assembly:
	g++ $< -g3 -S -c  pi.cpp -o assem.S $(library_FLAGS)


time:

clean:
	rm pi.out pi pi_serial
	
