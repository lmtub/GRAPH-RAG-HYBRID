static void tight_palette_rgb2buf(uint32_t rgb, int bpp, uint8_t buf[6])

{

    memset(buf, 0, 6);



    if (bpp == 32) {

        buf[0] = ((rgb >> 24) & 0xFF);

        buf[1] = ((rgb >> 16) & 0xFF);

        buf[2] = ((rgb >>  8) & 0xFF);

        buf[3] = ((rgb >>  0) & 0xFF);

        buf[4] = ((buf[0] & 1) == 0) << 3 | ((buf[1] & 1) == 0) << 2;

        buf[4]|= ((buf[2] & 1) == 0) << 1 | ((buf[3] & 1) == 0) << 0;

        buf[0] |= 1;

        buf[1] |= 1;

        buf[2] |= 1;

        buf[3] |= 1;

    }

    if (bpp == 16) {

        buf[0] = ((rgb >> 8) & 0xFF);

        buf[1] = ((rgb >> 0) & 0xFF);

        buf[2] = ((buf[0] & 1) == 0) << 1 | ((buf[1] & 1) == 0) << 0;

        buf[0] |= 1;

        buf[1] |= 1;

    }

}
