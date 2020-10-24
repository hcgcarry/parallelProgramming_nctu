// pi_multi_thread.c
#include <iostream>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#define SIMD_32Byte_NUM 8
#define Max_rand  4294967295
using namespace std;
uint32_t global_next(void) ;
uint32_t global_status[4] ={0};
typedef struct
{
   int thread_id;
   int start;
   int end;
   long long *totalCount;
   long long *totalHitCount;
} threadArgs; // 傳入 thread 的參數型別

static inline __m256i rotl(const __m256i x, int k)
{
   return _mm256_xor_si256(_mm256_slli_epi32(x, k), _mm256_srli_epi32(x, 32 - k));
}

//this should be function local variable
__m256i xorshift128plusplus_avx2(__m256i &state0, __m256i &state1, __m256i &state2, __m256i &state3)
{
   //const uint32_t result = rotl(s[0] + s[3], 7) + s[0];
   //unsigned may  need to check overflow
   __m256i result = _mm256_add_epi32(rotl(_mm256_add_epi32(state0, state3), 7), state0);

   //const uint32_t t = s[1] << 9;
   __m256i t = _mm256_slli_epi32(state1, 9);
   //s[2] ^= s[0];
   state2= _mm256_xor_si256(state2, state0);
   //s[3] ^= s[1];
   state3=_mm256_xor_si256(state3, state1);
   //s[1] ^= s[2];
   state1=_mm256_xor_si256(state1, state2);
   //s[0] ^= s[3];
   state0=_mm256_xor_si256(state0, state3);

   //s[2] ^= t;
   state2=_mm256_xor_si256(state2, t);

   //s[3] = rotl(s[3], 11);
   state3 = rotl(state3, 11);

   return result;
}

pthread_mutex_t sumMutex,global_status_mutex; // pthread 互斥鎖

// 每個 thread 要做的任務
__m256 caculateXorY(__m256i &randnum)
{
   __m256 float_randNum;
   unsigned int * randnum_ptr = (unsigned int *)&randnum;
   float* float_randNum_ptr= (float*)&float_randNum;
   for (int i = 0; i < SIMD_32Byte_NUM; i++)
   {
      //randnum_ptr[i] %= Max_rand;
      float_randNum_ptr[i] =  (float)randnum_ptr[i] ;
   }

   __m256 __v_float_2 = _mm256_set1_ps(2);
   __m256 __v_float_randmax = _mm256_set1_ps(Max_rand);
   __m256 __v_float_1 = _mm256_set1_ps(1);
   __m256 result = _mm256_sub_ps(\
       _mm256_mul_ps(\
           _mm256_div_ps(\
               float_randNum,
               __v_float_randmax),\
           __v_float_2),\
       __v_float_1);
   return result;
}
void printM256i(__m256i & avxValue)
{
   #ifdef debug
   int * tmp = (int*)&avxValue;
   for (int i = 0; i < SIMD_32Byte_NUM; i++)
   {
      cout << "i" << i << ":" << tmp[i] << " ";
   }
   cout << endl;
   #endif
}
void printM256unsinedInt(__m256i & avxValue)
{
   #ifdef debug
   unsigned int* tmp = (unsigned int*)&avxValue;
   for (int i = 0; i < SIMD_32Byte_NUM; i++)
   {
      cout << "i" << i << ":" << tmp[i] << " ";
   }
   cout << endl;
   #endif
}

void printM256(__m256 & avxValue)
{
   #ifdef debug
   float * tmp = (float*)&avxValue;
   for (int i = 0; i < SIMD_32Byte_NUM; i++)
   {
      cout << "i" << i << ":" << tmp[i] << " ";
   }
   cout << endl;
   #endif
}
///////////////// serial random
static inline uint32_t rotl_serial(const uint32_t x, int k)
{
   return (x << k) | (x >> (32 - k));
}
static uint32_t s_serial[4] ;
uint32_t next_s_serial(void)
{
   uint32_t res_serial = rotl_serial(s_serial[0] + s_serial[3], 7) + s_serial[0];
   const uint32_t t = s_serial[1] << 9;

   s_serial[2] ^= s_serial[0];
   s_serial[3] ^= s_serial[1];
   s_serial[1] ^= s_serial[2];
   s_serial[0] ^= s_serial[3];

   s_serial[2] ^= t;

   s_serial[3] = rotl_serial(s_serial[3], 11);
   res_serial %= Max_rand;

   return res_serial;
}
float caculateXorY_serial(unsigned int randnum){
  return (float)randnum*2/Max_rand-1;
   
}

void global_jump(void) {
   pthread_mutex_lock(&global_status_mutex);
	static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

	uint32_t s0 = 0;
	uint32_t s1 = 0;
	uint32_t s2 = 0;
	uint32_t s3 = 0;
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 32; b++) {
			if (JUMP[i] & UINT32_C(1) << b) {
				s0 ^= global_status[0];
				s1 ^= global_status[1];
				s2 ^= global_status[2];
				s3 ^= global_status[3];
			}
			global_next();	
		}
		
	global_status[0] = s0;
	global_status[1] = s1;
	global_status[2] = s2;
	global_status[3] = s3;
   pthread_mutex_unlock(&global_status_mutex);
}

uint32_t global_next(void) {
	const uint32_t result= rotl_serial(global_status[0] + global_status[3], 7) + global_status[0];

	const uint32_t t = global_status[1] << 9;

	global_status[2] ^= global_status[0];
	global_status[3] ^= global_status[1];
	global_status[1] ^= global_status[2];
	global_status[0] ^= global_status[3];

	global_status[2] ^= t;

	global_status[3] = rotl_serial(global_status[3], 11);

	return result;
}

void *count_pi(void *arg)
{
   //this line is important,it will prevent segmentation fault;
   //srand( time(NULL) );
   uint32_t *s =(unsigned int*)aligned_alloc(32,32*sizeof(uint32_t));

   for(int i=0;i<SIMD_32Byte_NUM;i++){
      for(int j=0;j<4;j++){
         s[i+SIMD_32Byte_NUM*j] =  global_status[j];
      }
      global_jump();
   }
   #ifdef debug
   cout << endl;
   for(int i=0;i<4;i++){
      cout <<  "statei" << i << endl;
      for(int j=0;j<8;j++){
         cout << s[i*8+j] << " ";
      }
      cout << endl;
   }
   cout << endl;
   #endif

   threadArgs *data = (threadArgs *)arg;
   int start = data->start;
   int end = data->end;
   long long *totalCount = data->totalCount;
   long long *totalHitCount = data->totalHitCount;
   long long local_hitCount = 0;

   __m256i curStatus[4];
   for (int i = 0; i < 4; i++)
   {
      curStatus[i] = _mm256_load_si256((const __m256i *)&s[i * SIMD_32Byte_NUM]);
      //__m256i curStatus[i]= _mm256_maskload_epi32((const int*)(s+i*8), maskAll);
      //curStatus[i]= _mm256_maskload_epi32((const int*)(s+i*8), maskAll);
      //cout << "curstatus i" << i  << endl;
      printM256i(curStatus[i]);
   }

   for (int toss = start; toss < end; toss += SIMD_32Byte_NUM) {
      #ifdef debug
      cout << "------------toss---------" << toss << endl;
      #endif
        /* Initialize the mask vector */
      //__m256i maskAll = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
      /* Selectively load data into the vector */
      //load state
      //status
      
      __m256i tmp_x = xorshift128plusplus_avx2(curStatus[0], curStatus[1], curStatus[2], curStatus[3]);
      __m256 x = caculateXorY(tmp_x);
      #ifdef debug
      cout << "#####vector_x###" << endl;

      cout << "v_tmp_x" << endl;
      printM256unsinedInt(tmp_x);
      cout << "x" << endl;
      printM256(x);
      //////////// serial
      cout << "#####serial_x###" << endl;
      cout << "serial_tmp_x" << endl;
      unsigned int serial_tmp_x = next_s_serial();
      cout << serial_tmp_x << endl;
      cout << "serial_x" << endl;
      cout << caculateXorY_serial(serial_tmp_x) << endl;

      //float y= (double)rand()*2/RAND_MAX -1;
      #endif

      __m256i tmp_y = xorshift128plusplus_avx2(curStatus[0], curStatus[1], curStatus[2], curStatus[3]);
      __m256 y = caculateXorY(tmp_y);
      #ifdef debug

      cout << "#####vector_y###" << endl;
      cout << "v_tmp_y" << endl;
      printM256unsinedInt(tmp_y);
      cout << "y" << endl;
      printM256(y);

      //////////// serial
      cout << "#####serial_y###" << endl;
      cout << "serial_tmp_y" << endl;
      unsigned int serial_tmp_y = next_s_serial();
      cout << serial_tmp_y << endl;
      cout << "serial_y" << endl;
      cout << caculateXorY_serial(serial_tmp_y) << endl;
      #endif

      //float distance_squared = x * x + y * y;
      __m256 distance_squared = _mm256_add_ps(_mm256_mul_ps(x, x), _mm256_mul_ps(y, y));
      //if ( distance_squared <= 1)
      //local_hitCount++;
      float* distance_squared_ptr = (float*)&distance_squared;
      for (int i = 0; i < SIMD_32Byte_NUM; i++)
      {
         if (distance_squared_ptr[i] <= 1)
            local_hitCount++;
      }
   }

   // **** 關鍵區域 ****
   // 一次只允許一個 thread 存取
   //printf("threadId:%d , totalCount:%d,totalHitCount:%d\n", data->thread_id, end - start, local_hitCount);
   pthread_mutex_lock(&sumMutex);
   *totalCount += (end - start);
   *totalHitCount += local_hitCount;
   pthread_mutex_unlock(&sumMutex);
   // *****************
   pthread_exit((void *)0);
}

int main(int argc, char *argv[])
{
   for(int i=0;i<4;i++){
      global_status[i] = 1+i;
   }
   int numThread = atoi(argv[1]);
   int count = atoi(argv[2]);
   pthread_t threads[numThread]; // 宣告建立 pthread
   // 初始化互斥鎖
   pthread_mutex_init(&sumMutex, NULL);
   pthread_mutex_init(&global_status_mutex, NULL);

   // 設定 pthread 性質是要能 join
   pthread_attr_t attr;
   pthread_attr_init(&attr);
   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

   // 每個 thread 都可以存取的 PI
   // 因為不同 thread 都要能存取，故用指標
   long long *totalCountPtr = (long long *)malloc(sizeof(int));
   long long *totalHitCountPtr = (long long *)malloc(sizeof(int));
   *totalCountPtr = 0;
   *totalHitCountPtr = 0;

   int part = count / numThread;
   threadArgs args[numThread]; // 每個 thread 傳入的參數
   for (int i = 0; i < numThread; i++)
   {
      // 設定傳入參數
      args[i].thread_id = i;
      args[i].start = part * i;
      args[i].end = part * (i + 1);
      args[i].totalCount = totalCountPtr;
      args[i].totalHitCount = totalHitCountPtr;

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
   float pi = 4 * (*totalHitCountPtr) / (double)(*totalCountPtr);
   printf("Pi =  %.10lf \n", pi);

   // 回收互斥鎖
   pthread_mutex_destroy(&sumMutex);
   pthread_mutex_destroy(&global_status_mutex);

   // 離開
   pthread_exit(NULL);
}