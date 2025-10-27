static bool logic_imm_decode_wmask(uint64_t *result, unsigned int immn,

                                   unsigned int imms, unsigned int immr)

{

    uint64_t mask;

    unsigned e, levels, s, r;

    int len;



    assert(immn < 2 && imms < 64 && immr < 64);



    /* The bit patterns we create here are 64 bit patterns which

     * are vectors of identical elements of size e = 2, 4, 8, 16, 32 or

     * 64 bits each. Each element contains the same value: a run

     * of between 1 and e-1 non-zero bits, rotated within the

     * element by between 0 and e-1 bits.

     *

     * The element size and run length are encoded into immn (1 bit)

     * and imms (6 bits) as follows:

     * 64 bit elements: immn = 1, imms = <length of run - 1>

     * 32 bit elements: immn = 0, imms = 0 : <length of run - 1>

     * 16 bit elements: immn = 0, imms = 10 : <length of run - 1>

     *  8 bit elements: immn = 0, imms = 110 : <length of run - 1>

     *  4 bit elements: immn = 0, imms = 1110 : <length of run - 1>

     *  2 bit elements: immn = 0, imms = 11110 : <length of run - 1>

     * Notice that immn = 0, imms = 11111x is the only combination

     * not covered by one of the above options; this is reserved.

     * Further, <length of run - 1> all-ones is a reserved pattern.

     *

     * In all cases the rotation is by immr % e (and immr is 6 bits).

     */



    /* First determine the element size */

    len = 31 - clz32((immn << 6) | (~imms & 0x3f));

    if (len < 1) {

        /* This is the immn == 0, imms == 0x11111x case */

        return false;

    }

    e = 1 << len;



    levels = e - 1;

    s = imms & levels;

    r = immr & levels;



    if (s == levels) {

        /* <length of run - 1> mustn't be all-ones. */

        return false;

    }



    /* Create the value of one element: s+1 set bits rotated

     * by r within the element (which is e bits wide)...

     */

    mask = bitmask64(s + 1);

    mask = (mask >> r) | (mask << (e - r));

    /* ...then replicate the element over the whole 64 bit value */

    mask = bitfield_replicate(mask, e);

    *result = mask;

    return true;

}
