static int tta_get_unary(GetBitContext *gb)

{

    int ret = 0;



    // count ones

    while(get_bits1(gb))

        ret++;

    return ret;

}
