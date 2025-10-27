static void pcx_palette(const uint8_t **src, uint32_t *dst, unsigned int pallen) {

    unsigned int i;



    for (i=0; i<pallen; i++)

        *dst++ = 0xFF000000 | bytestream_get_be24(src);

    if (pallen < 256)

        memset(dst, 0, (256 - pallen) * sizeof(*dst));

}
