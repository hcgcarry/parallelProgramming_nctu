// pi_multi_thread.c
#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
   int thread_id;
   int start;
   int end;
   long long* totalCount;
   long long* totalHitCount;
} threadArgs; // 傳入 thread 的參數型別


pthread_mutex_t sumMutex;    // pthread 互斥鎖

// 每個 thread 要做的任務
void *count_pi(void *arg)
{

   threadArgs *data = (threadArgs*)arg;
   int start = data->start;
   int end = data->end;
   long long* totalCount = data->totalCount;
   long long* totalHitCount = data->totalHitCount;
   long long local_hitCount = 0;
   
    
    for (int toss = start; toss < end; toss ++) {
                float x= (double)rand()*2/RAND_MAX -1;
                float y= (double)rand()*2/RAND_MAX -1;
                float distance_squared = x * x + y * y;
                     if ( distance_squared <= 1)
                        local_hitCount++;
    }

   // **** 關鍵區域 ****
   // 一次只允許一個 thread 存取
   printf("threadId:%d , totalCount:%d,totalHitCount:%d\n",data->thread_id,end-start,local_hitCount);
   pthread_mutex_lock(&sumMutex);
   *totalCount += (end-start);
   *totalHitCount += local_hitCount;
   pthread_mutex_unlock(&sumMutex);
   // *****************
   pthread_exit((void *)0);
}

int main(int argc, char *argv[])
{
    int numThread = atoi(argv[1]);
    int count = atoi(argv[2]);
    pthread_t threads[numThread]; // 宣告建立 pthread
   // 初始化互斥鎖
   pthread_mutex_init(&sumMutex, NULL);

   // 設定 pthread 性質是要能 join
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

   // 每個 thread 都可以存取的 PI
   // 因為不同 thread 都要能存取，故用指標
   long long* totalCountPtr = (long long*)malloc(sizeof(int));
   long long* totalHitCountPtr= (long long*)malloc(sizeof(int));
   *totalCountPtr= 0;
   *totalHitCountPtr = 0;

   int part = count/ numThread;
   threadArgs args[numThread]; // 每個 thread 傳入的參數
   for (int i = 0; i < numThread; i++)
   {
      // 設定傳入參數
      args[i].thread_id = i;
      args[i].start = part * i;
      args[i].end = part * (i + 1);
      args[i].totalCount= totalCountPtr;
      args[i].totalHitCount= totalHitCountPtr;

      // 建立一個 thread，執行 count_pi 任務，傳入 arg[i] 指標參數
      pthread_create(&threads[i], &attr, count_pi, (void *)&args[i]);
   }

   // 回收性質設定
   pthread_attr_destroy(&attr);

   void *status;
   for (int i = 0; i < numThread; i++)
   {
      // 等待每一個 thread 執行完畢
      pthread_join(threads[i], &status);
   }

   // 所有 thread 執行完畢，印出 PI
   float pi = 4*(*totalHitCountPtr)/(double)(*totalCountPtr);
   printf("Pi =  %.10lf \n", pi);

   // 回收互斥鎖
   pthread_mutex_destroy(&sumMutex);
   
   // 離開
   pthread_exit(NULL);
}