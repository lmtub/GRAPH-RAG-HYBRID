static inline int l1_unscale(int n, int mant, int scale_factor)

{

    int shift, mod;

    int64_t val;



    shift   = scale_factor_modshift[scale_factor];

    mod     = shift & 3;

    shift >>= 2;

    val     = MUL64(mant + (-1 << n) + 1, scale_factor_mult[n-1][mod]);

    shift  += n;

    /* NOTE: at this point, 1 <= shift >= 21 + 15 */

    return (int)((val + (1LL << (shift - 1))) >> shift);

}
