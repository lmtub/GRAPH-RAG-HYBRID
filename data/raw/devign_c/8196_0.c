static inline int compress_coeffs(int *coef, int order, int c_bits)

{

    int i, res = 0;

    const int low_idx   = c_bits ?  4 : 2;

    const int shift_val = c_bits ?  8 : 4;

    const int high_idx  = c_bits ? 11 : 5;

    for (i = 0; i < order; i++)

        if (coef[i] < low_idx || coef[i] > high_idx)

            res++;

    if (res == order)

        for (i = 0; i < order; i++)

            coef[i] -= (coef[i] > high_idx) ? shift_val : 0;

    return res == order;

}
