uint64_t hbitmap_serialization_granularity(const HBitmap *hb)

{

    /* Must hold true so that the shift below is defined

     * (ld(64) == 6, i.e. 1 << 6 == 64) */

    assert(hb->granularity < 64 - 6);



    /* Require at least 64 bit granularity to be safe on both 64 bit and 32 bit

     * hosts. */

    return UINT64_C(64) << hb->granularity;

}
