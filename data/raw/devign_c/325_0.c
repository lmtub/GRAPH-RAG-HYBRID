static uint32_t tight_palette_buf2rgb(int bpp, const uint8_t *buf)

{

    uint32_t rgb = 0;



    if (bpp == 32) {

        rgb |= ((buf[0] & ~1) | !((buf[4] >> 3) & 1)) << 24;

        rgb |= ((buf[1] & ~1) | !((buf[4] >> 2) & 1)) << 16;

        rgb |= ((buf[2] & ~1) | !((buf[4] >> 1) & 1)) <<  8;

        rgb |= ((buf[3] & ~1) | !((buf[4] >> 0) & 1)) <<  0;

    }

    if (bpp == 16) {

        rgb |= ((buf[0] & ~1) | !((buf[2] >> 1) & 1)) << 8;

        rgb |= ((buf[1] & ~1) | !((buf[2] >> 0) & 1)) << 0;

    }

    return rgb;

}
