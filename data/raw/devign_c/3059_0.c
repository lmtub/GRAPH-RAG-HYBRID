uint64_t HELPER(cvd)(int32_t bin)

{

    /* positive 0 */

    uint64_t dec = 0x0c;

    int shift = 4;



    if (bin < 0) {

        bin = -bin;

        dec = 0x0d;

    }



    for (shift = 4; (shift < 64) && bin; shift += 4) {

        int current_number = bin % 10;



        dec |= (current_number) << shift;

        bin /= 10;

    }



    return dec;

}
