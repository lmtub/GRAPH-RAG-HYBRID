static int32_t scalarproduct_and_madd_int32_c(int16_t *v1, const int32_t *v2,

                                              const int16_t *v3,

                                              int order, int mul)

{

    int res = 0;



    while (order--) {

        res   += *v1 * *v2++;

        *v1++ += mul * *v3++;

    }

    return res;

}
