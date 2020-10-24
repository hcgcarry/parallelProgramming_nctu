#include<iostream>
#include<cstdlib>
using namespace std;




int main(void){
    long long number_in_circle =0;
    
    
    
    
    long long number_of_tosses =(long long) 1 << 29;
    
    cout << "number of tosses " <<number_of_tosses << endl;

    
    for (int toss = 0; toss < number_of_tosses; toss ++) {
                float x= (double)rand()*2/RAND_MAX -1;
                float y= (double)rand()*2/RAND_MAX -1;
                float distance_squared = x * x + y * y;
                
                        if ( distance_squared <= 1)
                                    number_in_circle++;
    }
    float pi_estimate = 4 * number_in_circle /(( double ) number_of_tosses);
    cout << pi_estimate << endl;

}


