static inline int l3_unscale(int value, int exponent)

{

    unsigned int m;

    int e;



    e  = table_4_3_exp  [4 * value + (exponent & 3)];

    m  = table_4_3_value[4 * value + (exponent & 3)];

    e -= exponent >> 2;

#ifdef DEBUG

    if(e < 1)

        av_log(NULL, AV_LOG_WARNING, "l3_unscale: e is %d\n", e);

#endif

    if (e > (SUINT)31)

        return 0;

    m = (m + (1 << (e - 1))) >> e;



    return m;

}
