#include "PPintrin.h"
#include "logger.h"
#include <iostream>
using namespace std;
//******************
//* Implementation *
//******************
#include "logger.h"

__pp_mask _pp_init_ones(int first)
{
  __pp_mask mask;
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    mask.value[i] = (i < first) ? true : false;
  }
  return mask;
}

__pp_mask _pp_mask_not(__pp_mask &maska)
{
  
  cout << "--------mask not----------" << endl;
  printMask(maska);
  __pp_mask resultMask;
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    resultMask.value[i] = !maska.value[i];
  }
  PPLogger.addLog("masknot", _pp_init_ones(), VECTOR_WIDTH);
  printMask(resultMask);
  
  return resultMask;
}

__pp_mask _pp_mask_or(__pp_mask &maska, __pp_mask &maskb)
{
  cout << "-----------mask or--------" << endl;
  __pp_mask resultMask;
  printMask(maskb);
  printMask(maska);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    resultMask.value[i] = maska.value[i] | maskb.value[i];
  }
  PPLogger.addLog("maskor", _pp_init_ones(), VECTOR_WIDTH);
  printMask(resultMask);
  return resultMask;
}

__pp_mask _pp_mask_and(__pp_mask &maska, __pp_mask &maskb)
{
  cout << "-----------mask and--------" << endl;
  printMask(maskb);
  printMask(maska);
  __pp_mask resultMask;
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    resultMask.value[i] = maska.value[i] && maskb.value[i];
  }
  PPLogger.addLog("maskand", _pp_init_ones(), VECTOR_WIDTH);
  printMask(resultMask);
  return resultMask;
}

int _pp_cntbits(__pp_mask &maska)
{
  cout << "-------cntbits----" << endl;
  printMask(maska);
  int count = 0;
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    if (maska.value[i])
      count++;
  }
  PPLogger.addLog("cntbits", _pp_init_ones(), VECTOR_WIDTH);
  cout << "return count" << count << endl;
  return count;
}

template <typename T>
void _pp_vset(__pp_vec<T> &vecResult, T value, __pp_mask &mask)
{
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    vecResult.value[i] = mask.value[i] ? value : vecResult.value[i];
  }
  PPLogger.addLog("vset", mask, VECTOR_WIDTH);
}

template void _pp_vset<float>(__pp_vec_float &vecResult, float value, __pp_mask &mask);
template void _pp_vset<int>(__pp_vec_int &vecResult, int value, __pp_mask &mask);

void _pp_vset_float(__pp_vec_float &vecResult, float value, __pp_mask &mask) { _pp_vset<float>(vecResult, value, mask); }
void _pp_vset_int(__pp_vec_int &vecResult, int value, __pp_mask &mask) { _pp_vset<int>(vecResult, value, mask); }

__pp_vec_float _pp_vset_float(float value)
{
  cout << "-------vset------" << endl;
  __pp_vec_float vecResult;
  __pp_mask mask = _pp_init_ones();
  _pp_vset_float(vecResult, value, mask);
  printFloatVar(vecResult);
  return vecResult;
}
__pp_vec_int _pp_vset_int(int value)
{
  cout << "-----------vset-------" << endl;
  __pp_vec_int vecResult;
  __pp_mask mask = _pp_init_ones();
  _pp_vset_int(vecResult, value, mask);
  printIntVar(vecResult);
  return vecResult;
}

template <typename T>
void _pp_vmove(__pp_vec<T> &dest, __pp_vec<T> &src, __pp_mask &mask)
{
  printMask(mask);
  printVar(src);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    dest.value[i] = mask.value[i] ? src.value[i] : dest.value[i];
  }
  printVar(dest);
  PPLogger.addLog("vmove", mask, VECTOR_WIDTH);
}

template void _pp_vmove<float>(__pp_vec_float &dest, __pp_vec_float &src, __pp_mask &mask);
template void _pp_vmove<int>(__pp_vec_int &dest, __pp_vec_int &src, __pp_mask &mask);

void _pp_vmove_float(__pp_vec_float &dest, __pp_vec_float &src, __pp_mask &mask) { _pp_vmove<float>(dest, src, mask); }
void _pp_vmove_int(__pp_vec_int &dest, __pp_vec_int &src, __pp_mask &mask) { _pp_vmove<int>(dest, src, mask); }

template <typename T>
void _pp_vload(__pp_vec<T> &dest, T *src, __pp_mask &mask)
{
  cout << "-----------vload------------" << endl;
  printMask(mask);
  printArray(src);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    dest.value[i] = mask.value[i] ? src[i] : dest.value[i];
  }
  printVar(dest);
  PPLogger.addLog("vload", mask, VECTOR_WIDTH);
}

template void _pp_vload<float>(__pp_vec_float &dest, float *src, __pp_mask &mask);
template void _pp_vload<int>(__pp_vec_int &dest, int *src, __pp_mask &mask);

void _pp_vload_float(__pp_vec_float &dest, float *src, __pp_mask &mask) { _pp_vload<float>(dest, src, mask); }
void _pp_vload_int(__pp_vec_int &dest, int *src, __pp_mask &mask) { _pp_vload<int>(dest, src, mask); }

template <typename T>
void _pp_vstore(T *dest, __pp_vec<T> &src, __pp_mask &mask)
{
  cout << "-----------vstore--------" << endl;
  printMask(mask);
  printVar(src);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    dest[i] = mask.value[i] ? src.value[i] : dest[i];
  }
  printArray(dest);
  PPLogger.addLog("vstore", mask, VECTOR_WIDTH);
}

template void _pp_vstore<float>(float *dest, __pp_vec_float &src, __pp_mask &mask);
template void _pp_vstore<int>(int *dest, __pp_vec_int &src, __pp_mask &mask);

void _pp_vstore_float(float *dest, __pp_vec_float &src, __pp_mask &mask) { _pp_vstore<float>(dest, src, mask); }
void _pp_vstore_int(int *dest, __pp_vec_int &src, __pp_mask &mask) { _pp_vstore<int>(dest, src, mask); }

template <typename T>
void _pp_vadd(__pp_vec<T> &vecResult, __pp_vec<T> &veca, __pp_vec<T> &vecb, __pp_mask &mask)
{
  cout << "--------vadd--------" << endl;
  printMask(mask);
  printVar(vecb);
  printVar(veca);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    vecResult.value[i] = mask.value[i] ? (veca.value[i] + vecb.value[i]) : vecResult.value[i];
  }
  printVar(vecResult);
  
  PPLogger.addLog("vadd", mask, VECTOR_WIDTH);
}

template void _pp_vadd<float>(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
template void _pp_vadd<int>(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

void _pp_vadd_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask) { _pp_vadd<float>(vecResult, veca, vecb, mask); }
void _pp_vadd_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask) { _pp_vadd<int>(vecResult, veca, vecb, mask); }

template <typename T>
void _pp_vsub(__pp_vec<T> &vecResult, __pp_vec<T> &veca, __pp_vec<T> &vecb, __pp_mask &mask)
{
  cout << "-----vsub-------" << endl;
  printMask(mask);
  printVar(vecb);
  printVar(veca);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    vecResult.value[i] = mask.value[i] ? (veca.value[i] - vecb.value[i]) : vecResult.value[i];
  }
  printVar(vecResult);
  PPLogger.addLog("vsub", mask, VECTOR_WIDTH);
}

template void _pp_vsub<float>(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
template void _pp_vsub<int>(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

void _pp_vsub_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask) { _pp_vsub<float>(vecResult, veca, vecb, mask); }
void _pp_vsub_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask) { _pp_vsub<int>(vecResult, veca, vecb, mask); }

template <typename T>
void _pp_vmult(__pp_vec<T> &vecResult, __pp_vec<T> &veca, __pp_vec<T> &vecb, __pp_mask &mask)
{
  cout << "-----vmult-------" << endl;
  printMask(mask);
  printVar(vecb);
  printVar(veca);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    vecResult.value[i] = mask.value[i] ? (veca.value[i] * vecb.value[i]) : vecResult.value[i];
  }
  printVar(vecResult);
  PPLogger.addLog("vmult", mask, VECTOR_WIDTH);
}

template void _pp_vmult<float>(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
template void _pp_vmult<int>(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

void _pp_vmult_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask) { _pp_vmult<float>(vecResult, veca, vecb, mask); }
void _pp_vmult_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask) { _pp_vmult<int>(vecResult, veca, vecb, mask); }

template <typename T>
void _pp_vdiv(__pp_vec<T> &vecResult, __pp_vec<T> &veca, __pp_vec<T> &vecb, __pp_mask &mask)
{
  cout << "-----vdiv-------" << endl;
  printMask(mask);
  printVar(vecb);
  printVar(veca);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    vecResult.value[i] = mask.value[i] ? (veca.value[i] / vecb.value[i]) : vecResult.value[i];
  }
  printVar(vecResult);
  PPLogger.addLog("vdiv", mask, VECTOR_WIDTH);
}

template void _pp_vdiv<float>(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
template void _pp_vdiv<int>(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

void _pp_vdiv_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask) { _pp_vdiv<float>(vecResult, veca, vecb, mask); }
void _pp_vdiv_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask) { _pp_vdiv<int>(vecResult, veca, vecb, mask); }

template <typename T>
void _pp_vabs(__pp_vec<T> &vecResult, __pp_vec<T> &veca, __pp_mask &mask)
{
  cout << "--------vabs-------" << endl;
  printMask(mask);
  printVar(veca);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    vecResult.value[i] = mask.value[i] ? (abs(veca.value[i])) : vecResult.value[i];
  }
  printVar(vecResult);
  PPLogger.addLog("vabs", mask, VECTOR_WIDTH);
}

template void _pp_vabs<float>(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_mask &mask);
template void _pp_vabs<int>(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_mask &mask);

void _pp_vabs_float(__pp_vec_float &vecResult, __pp_vec_float &veca, __pp_mask &mask) { _pp_vabs<float>(vecResult, veca, mask); }
void _pp_vabs_int(__pp_vec_int &vecResult, __pp_vec_int &veca, __pp_mask &mask) { _pp_vabs<int>(vecResult, veca, mask); }

template <typename T>
void _pp_vgt(__pp_mask &maskResult, __pp_vec<T> &veca, __pp_vec<T> &vecb, __pp_mask &mask)
{
  cout << "--------vgt-------" << endl;
  printMask(mask);
  printVar(vecb);
  printVar(veca);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    maskResult.value[i] = mask.value[i] ? (veca.value[i] > vecb.value[i]) : maskResult.value[i];
  }
  printMask(maskResult);
  PPLogger.addLog("vgt", mask, VECTOR_WIDTH);
}

template void _pp_vgt<float>(__pp_mask &maskResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
template void _pp_vgt<int>(__pp_mask &maskResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

void _pp_vgt_float(__pp_mask &maskResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask) { _pp_vgt<float>(maskResult, veca, vecb, mask); }
void _pp_vgt_int(__pp_mask &maskResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask) { _pp_vgt<int>(maskResult, veca, vecb, mask); }

template <typename T>
void _pp_vlt(__pp_mask &maskResult, __pp_vec<T> &veca, __pp_vec<T> &vecb, __pp_mask &mask)
{
  cout << "--------vlt-------" << endl;
  printMask(mask);
  printVar(vecb);
  printVar(veca);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    maskResult.value[i] = mask.value[i] ? (veca.value[i] < vecb.value[i]) : maskResult.value[i];
  }
  printMask(maskResult);
  PPLogger.addLog("vlt", mask, VECTOR_WIDTH);
}

template void _pp_vlt<float>(__pp_mask &maskResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
template void _pp_vlt<int>(__pp_mask &maskResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

void _pp_vlt_float(__pp_mask &maskResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask) { _pp_vlt<float>(maskResult, veca, vecb, mask); }
void _pp_vlt_int(__pp_mask &maskResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask) { _pp_vlt<int>(maskResult, veca, vecb, mask); }

template <typename T>
void _pp_veq(__pp_mask &maskResult, __pp_vec<T> &veca, __pp_vec<T> &vecb, __pp_mask &mask)
{
  cout << "--------veq-------" << endl;
  printMask(mask);
  printVar(vecb);
  printVar(veca);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    maskResult.value[i] = mask.value[i] ? (veca.value[i] == vecb.value[i]) : maskResult.value[i];
  }
  printMask(maskResult);
  PPLogger.addLog("veq", mask, VECTOR_WIDTH);
}

template void _pp_veq<float>(__pp_mask &maskResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask);
template void _pp_veq<int>(__pp_mask &maskResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask);

void _pp_veq_float(__pp_mask &maskResult, __pp_vec_float &veca, __pp_vec_float &vecb, __pp_mask &mask) { _pp_veq<float>(maskResult, veca, vecb, mask); }
void _pp_veq_int(__pp_mask &maskResult, __pp_vec_int &veca, __pp_vec_int &vecb, __pp_mask &mask) { _pp_veq<int>(maskResult, veca, vecb, mask); }

template <typename T>
void _pp_hadd(__pp_vec<T> &vecResult, __pp_vec<T> &vec)
{
  cout << "--------hadd-------" << endl;
  printVar(vec);
  for (int i = 0; i < VECTOR_WIDTH / 2; i++)
  {
    T result = vec.value[2 * i] + vec.value[2 * i + 1];
    vecResult.value[2 * i] = result;
    vecResult.value[2 * i + 1] = result;
  }
  printVar(vecResult);
}

template void _pp_hadd<float>(__pp_vec_float &vecResult, __pp_vec_float &vec);

void _pp_hadd_float(__pp_vec_float &vecResult, __pp_vec_float &vec) { _pp_hadd<float>(vecResult, vec); }

template <typename T>
void _pp_interleave(__pp_vec<T> &vecResult, __pp_vec<T> &vec)
{
  cout << "---------interleave--------" << endl;
  printVar(vec);
  for (int i = 0; i < VECTOR_WIDTH; i++)
  {
    int index = i < VECTOR_WIDTH / 2 ? (2 * i) : (2 * (i - VECTOR_WIDTH / 2) + 1);
    vecResult.value[i] = vec.value[index];
  }
  printVar(vecResult);
}

template void _pp_interleave<float>(__pp_vec_float &vecResult, __pp_vec_float &vec);

void _pp_interleave_float(__pp_vec_float &vecResult, __pp_vec_float &vec) { _pp_interleave<float>(vecResult, vec); }

void addUserLog(const char *logStr)
{
  PPLogger.addLog(logStr, _pp_init_ones(), 0);
}

void printIntVar( __pp_vec<int>& var){
  //cout << varName << ":";
  for(int i=0;i<VECTOR_WIDTH;i++){
    cout << var.value[i] << " " ;
  }
  cout << endl;
}

void printFloatVar( __pp_vec<float>& var){
  //cout << varName << ":";
  for(int i=0;i<VECTOR_WIDTH;i++){
    cout << var.value[i] << " " ;
  }
  cout << endl;
}

void printMask(__pp_mask& mask){
  int result=0;
  for (int i = 0; i <VECTOR_WIDTH; i++)
  {
    cout << mask.value[i] << " ";
  }
  cout << endl;
}

