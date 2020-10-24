#include "PPintrin.h"

float addVector_Width(__pp_vec_float& vec_input,int vector_width);
// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float zero_float_v = _pp_vset_float(0.f);
  __pp_vec_float one_float_v = _pp_vset_float(1);
  __pp_vec_int zero_int_v = _pp_vset_int(0);
  __pp_vec_int one_int_v = _pp_vset_int(1);
  __pp_vec_float nine_float_v = _pp_vset_float(9.999999f);
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    //cout << "-----------interation----------" << i << endl;
    //initial
    __pp_vec_int y;
    __pp_vec_float x;
    __pp_vec_float result;
    __pp_vec_int count;
    int operandNum = VECTOR_WIDTH;
    if(i+VECTOR_WIDTH > N){
      operandNum = N % VECTOR_WIDTH;
    }
    __pp_mask maskAll= _pp_init_ones(operandNum),\
    mask_count_gt_9= _pp_init_ones(0),\
    mask_count_gt_0= _pp_init_ones(0),\
    mask_result_gt_9= _pp_init_ones(0),\
    mask_y_eq_zero= _pp_init_ones(0)\
    ,mask_y_neq_zero= _pp_init_ones(0) ;

    // float x = values[i];
    _pp_vload_float(x, values + i, maskAll); 
    
    //int y = exponents[i];
    _pp_vload_int(y, exponents + i, maskAll); 

    //if (y == 0)
    _pp_veq_int(mask_y_eq_zero,y, zero_int_v, maskAll); 

    // output[i] = 1.f;
    _pp_vstore_float(output+i,one_float_v,mask_y_eq_zero);

    //else
    mask_y_neq_zero = _pp_mask_not(mask_y_eq_zero);
    mask_y_neq_zero = _pp_mask_and(mask_y_neq_zero,maskAll);

  //      float result = x;
    _pp_vadd_float(result,zero_float_v,x,mask_y_neq_zero);
      //int count = y - 1;
    _pp_vsub_int(count,y,one_int_v,mask_y_neq_zero);
      //while (count > 0)
    _pp_vgt_int(mask_count_gt_0,count,zero_int_v,mask_y_neq_zero);
    while(_pp_cntbits(mask_count_gt_0)){
        //result *= x;
      _pp_vmult_float(result,result,x,mask_count_gt_0);
        //count--;
      _pp_vsub_int(count,count,one_int_v,mask_count_gt_0);
      _pp_vgt_int(mask_count_gt_0,count,zero_int_v,mask_count_gt_0);
    }

      //if (result > 9.999999f)
    _pp_vgt_float(mask_result_gt_9,result,nine_float_v,mask_y_neq_zero);
        //result = 9.999999f;
    _pp_vset_float(result,9.999999f,mask_result_gt_9);

    // Execute instruction using mask ("if" clause)
      //output[i] = result;
    _pp_vstore_float(output+i,result,mask_y_neq_zero);

  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  /*
  float sum=0;
  for(int i=0;i<N;i++){
    //cout << values[i] << " ";
    sum +=values[i];
  }
  */
  //cout << endl;
  //cout << "sum " << sum << endl;

  //float *afterAddVector_width_result=new float[afterAddVector_width_result_len];
  //float *afterAddVector_width_result =(float*)malloc(afterAddVector_width_result_len);
  __pp_mask maskAll = _pp_init_ones();
  __pp_vec_float vec_input_float_sum = _pp_vset_float(0);
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    __pp_vec_float vec_input_float_tmp;
    _pp_vload_float(vec_input_float_tmp,values+i,maskAll);
    _pp_vadd_float(vec_input_float_sum,vec_input_float_sum,vec_input_float_tmp,maskAll);
    //afterAddVector_width_result[i/VECTOR_WIDTH] = addVector_Width(vec_input_float,VECTOR_WIDTH);
  }
  return addVector_Width(vec_input_float_sum,VECTOR_WIDTH);


}


float addVector_Width(__pp_vec_float& vec_input,int vector_width){
  while(vector_width > 1){
    //cout << "vector_width " << vector_width << endl;
    _pp_hadd_float(vec_input,vec_input);
    _pp_interleave_float(vec_input,vec_input);
    vector_width /= 2;
  }
  //cout << " return result  " << vec_input.value[0] << endl;
  return vec_input.value[0];
}