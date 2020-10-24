#include<iostream>
#include <getopt.h>
#include<vector>
#include"test.h"
#include<algorithm>

using namespace std;

int main(int argc,char*argv[]){
    int opt;
    int type=1;

    static struct option long_options[] = {
        {"type", 1, 0, 't'},
        {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "t:", long_options, NULL)) != EOF)
    {
        if(opt == 't'){
            type = atoi(optarg);
        }
    }
    if(type==1){
        int index = 100;
        float a[1024],b[1024],c[1024];
        for(int i=0;i<1024;i++){
            a[i] = i;
            b[i] = i;
            c[i] = i;
        }
        vector<double> time_list(index,0);
        for(int i=0;i<index;i++){
            time_list[i] = test1(a,b,c);
        }
        sort(time_list.begin(),time_list.end());
        cout << "median of exe time" <<  time_list[index/2] << endl;
    }




}