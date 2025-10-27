static inline int l3_unscale(int value, int exponent)

{

    unsigned int m;

    int e;



    e  = table_4_3_exp  [4 * value + (exponent & 3)];

    m  = table_4_3_value[4 * value + (exponent & 3)];

    e -= exponent >> 2;

    assert(e >= 1);

    if (e > 31)

        return 0;

    m = (m + (1 << (e - 1))) >> e;



    return m;

}
