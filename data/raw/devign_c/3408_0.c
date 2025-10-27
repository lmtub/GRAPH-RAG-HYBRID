int sws_setColorspaceDetails(SwsContext *c, const int inv_table[4], int srcRange, const int table[4], int dstRange, int brightness, int contrast, int saturation)

{

    int64_t crv =  inv_table[0];

    int64_t cbu =  inv_table[1];

    int64_t cgu = -inv_table[2];

    int64_t cgv = -inv_table[3];

    int64_t cy  = 1<<16;

    int64_t oy  = 0;



    memcpy(c->srcColorspaceTable, inv_table, sizeof(int)*4);

    memcpy(c->dstColorspaceTable,     table, sizeof(int)*4);



    c->brightness= brightness;

    c->contrast  = contrast;

    c->saturation= saturation;

    c->srcRange  = srcRange;

    c->dstRange  = dstRange;

    if (isYUV(c->dstFormat) || isGray(c->dstFormat)) return -1;



    c->uOffset=   0x0400040004000400LL;

    c->vOffset=   0x0400040004000400LL;



    if (!srcRange) {

        cy= (cy*255) / 219;

        oy= 16<<16;

    } else {

        crv= (crv*224) / 255;

        cbu= (cbu*224) / 255;

        cgu= (cgu*224) / 255;

        cgv= (cgv*224) / 255;

    }



    cy = (cy *contrast             )>>16;

    crv= (crv*contrast * saturation)>>32;

    cbu= (cbu*contrast * saturation)>>32;

    cgu= (cgu*contrast * saturation)>>32;

    cgv= (cgv*contrast * saturation)>>32;



    oy -= 256*brightness;



    c->yCoeff=    roundToInt16(cy *8192) * 0x0001000100010001ULL;

    c->vrCoeff=   roundToInt16(crv*8192) * 0x0001000100010001ULL;

    c->ubCoeff=   roundToInt16(cbu*8192) * 0x0001000100010001ULL;

    c->vgCoeff=   roundToInt16(cgv*8192) * 0x0001000100010001ULL;

    c->ugCoeff=   roundToInt16(cgu*8192) * 0x0001000100010001ULL;

    c->yOffset=   roundToInt16(oy *   8) * 0x0001000100010001ULL;



    c->yuv2rgb_y_coeff  = (int16_t)roundToInt16(cy <<13);

    c->yuv2rgb_y_offset = (int16_t)roundToInt16(oy << 9);

    c->yuv2rgb_v2r_coeff= (int16_t)roundToInt16(crv<<13);

    c->yuv2rgb_v2g_coeff= (int16_t)roundToInt16(cgv<<13);

    c->yuv2rgb_u2g_coeff= (int16_t)roundToInt16(cgu<<13);

    c->yuv2rgb_u2b_coeff= (int16_t)roundToInt16(cbu<<13);



    ff_yuv2rgb_c_init_tables(c, inv_table, srcRange, brightness, contrast, saturation);

    //FIXME factorize



#if ARCH_PPC && HAVE_ALTIVEC

    if (c->flags & SWS_CPU_CAPS_ALTIVEC)

        ff_yuv2rgb_init_tables_altivec(c, inv_table, brightness, contrast, saturation);

#endif

    return 0;

}
