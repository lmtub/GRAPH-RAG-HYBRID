static int get_bitmap(QEMUFile *f, void *pv, size_t size)

{

    unsigned long *bmp = pv;

    int i, idx = 0;

    for (i = 0; i < BITS_TO_U64S(size); i++) {

        uint64_t w = qemu_get_be64(f);

        bmp[idx++] = w;

        if (sizeof(unsigned long) == 4 && idx < BITS_TO_LONGS(size)) {

            bmp[idx++] = w >> 32;

        }

    }

    return 0;

}
