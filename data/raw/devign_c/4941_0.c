static void xenfb_copy_mfns(int mode, int count, unsigned long *dst, void *src)

{

    uint32_t *src32 = src;

    uint64_t *src64 = src;

    int i;



    for (i = 0; i < count; i++)

	dst[i] = (mode == 32) ? src32[i] : src64[i];

}
