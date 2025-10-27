static const uint8_t *pcx_rle_decode(const uint8_t *src, uint8_t *dst,

                            unsigned int bytes_per_scanline, int compressed) {

    unsigned int i = 0;

    unsigned char run, value;



    if (compressed) {

        while (i<bytes_per_scanline) {

            run = 1;

            value = *src++;

            if (value >= 0xc0) {

                run = value & 0x3f;

                value = *src++;

            }

            while (i<bytes_per_scanline && run--)

                dst[i++] = value;

        }

    } else {

        memcpy(dst, src, bytes_per_scanline);

        src += bytes_per_scanline;

    }



    return src;

}
