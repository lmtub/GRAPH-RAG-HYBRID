static inline void conv_to_float(float *arr, int32_t *cof, int num)

{

    int i;

    for (i = 0; i < num; i++)

        arr[i] = (float)cof[i]/INT32_MAX;

}
