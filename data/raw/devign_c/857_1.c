void do_fctiw (void)

{

    union {

        double d;

        uint64_t i;

    } p;



    /* XXX: higher bits are not supposed to be significant.

     *     to make tests easier, return the same as a real PowerPC 750 (aka G3)

     */

    p.i = float64_to_int32(FT0, &env->fp_status);

    p.i |= 0xFFF80000ULL << 32;

    FT0 = p.d;

}
