static void mct_decode(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile)
{
    int i, csize = 1;
    int32_t *src[3],  i0,  i1,  i2;
    float   *srcf[3], i0f, i1f, i2f;
    for (i = 0; i < 3; i++)
        if (tile->codsty[0].transform == FF_DWT97)
            srcf[i] = tile->comp[i].f_data;
        else
            src [i] = tile->comp[i].i_data;
    for (i = 0; i < 2; i++)
        csize *= tile->comp[0].coord[i][1] - tile->comp[0].coord[i][0];
    switch (tile->codsty[0].transform) {
    case FF_DWT97:
        for (i = 0; i < csize; i++) {
            i0f = *srcf[0] + (f_ict_params[0] * *srcf[2]);
            i1f = *srcf[0] - (f_ict_params[1] * *srcf[1])
                           - (f_ict_params[2] * *srcf[2]);
            i2f = *srcf[0] + (f_ict_params[3] * *srcf[1]);
            *srcf[0]++ = i0f;
            *srcf[1]++ = i1f;
            *srcf[2]++ = i2f;
        break;
    case FF_DWT97_INT:
        for (i = 0; i < csize; i++) {
            i0 = *src[0] + (((i_ict_params[0] * *src[2]) + (1 << 15)) >> 16);
            i1 = *src[0] - (((i_ict_params[1] * *src[1]) + (1 << 15)) >> 16)
                         - (((i_ict_params[2] * *src[2]) + (1 << 15)) >> 16);
            i2 = *src[0] + (((i_ict_params[3] * *src[1]) + (1 << 15)) >> 16);
            *src[0]++ = i0;
            *src[1]++ = i1;
            *src[2]++ = i2;
        break;
    case FF_DWT53:
        for (i = 0; i < csize; i++) {
            i1 = *src[0] - (*src[2] + *src[1] >> 2);
            i0 = i1 + *src[2];
            i2 = i1 + *src[1];
            *src[0]++ = i0;
            *src[1]++ = i1;
            *src[2]++ = i2;
        break;