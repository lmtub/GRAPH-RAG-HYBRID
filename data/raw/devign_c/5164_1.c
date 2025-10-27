static void qpeg_decode_inter(const uint8_t *src, uint8_t *dst, int size,

                            int stride, int width, int height,

                            int delta, const uint8_t *ctable, uint8_t *refdata)

{

    int i, j;

    int code;

    int filled = 0;

    int orig_height;



    if(!refdata)

        refdata= dst;



    /* copy prev frame */

    for(i = 0; i < height; i++)

        memcpy(dst + (i * stride), refdata + (i * stride), width);



    orig_height = height;

    height--;

    dst = dst + height * stride;



    while((size > 0) && (height >= 0)) {

        code = *src++;

        size--;



        if(delta) {

            /* motion compensation */

            while((code & 0xF0) == 0xF0) {

                if(delta == 1) {

                    int me_idx;

                    int me_w, me_h, me_x, me_y;

                    uint8_t *me_plane;

                    int corr, val;



                    /* get block size by index */

                    me_idx = code & 0xF;

                    me_w = qpeg_table_w[me_idx];

                    me_h = qpeg_table_h[me_idx];



                    /* extract motion vector */

                    corr = *src++;

                    size--;



                    val = corr >> 4;

                    if(val > 7)

                        val -= 16;

                    me_x = val;



                    val = corr & 0xF;

                    if(val > 7)

                        val -= 16;

                    me_y = val;



                    /* check motion vector */

                    if ((me_x + filled < 0) || (me_x + me_w + filled > width) ||

                       (height - me_y - me_h < 0) || (height - me_y > orig_height) ||

                       (filled + me_w > width) || (height - me_h < 0))

                        av_log(NULL, AV_LOG_ERROR, "Bogus motion vector (%i,%i), block size %ix%i at %i,%i\n",

                               me_x, me_y, me_w, me_h, filled, height);

                    else {

                        /* do motion compensation */

                        me_plane = refdata + (filled + me_x) + (height - me_y) * stride;

                        for(j = 0; j < me_h; j++) {

                            for(i = 0; i < me_w; i++)

                                dst[filled + i - (j * stride)] = me_plane[i - (j * stride)];

                        }

                    }

                }

                code = *src++;

                size--;

            }

        }



        if(code == 0xE0) /* end-of-picture code */

            break;

        if(code > 0xE0) { /* run code: 0xE1..0xFF */

            int p;



            code &= 0x1F;

            p = *src++;

            size--;

            for(i = 0; i <= code; i++) {

                dst[filled++] = p;

                if(filled >= width) {

                    filled = 0;

                    dst -= stride;

                    height--;

                    if(height < 0)

                        break;

                }

            }

        } else if(code >= 0xC0) { /* copy code: 0xC0..0xDF */

            code &= 0x1F;



            for(i = 0; i <= code; i++) {

                dst[filled++] = *src++;

                if(filled >= width) {

                    filled = 0;

                    dst -= stride;

                    height--;

                    if(height < 0)

                        break;

                }

            }

            size -= code + 1;

        } else if(code >= 0x80) { /* skip code: 0x80..0xBF */

            int skip;



            code &= 0x3F;

            /* codes 0x80 and 0x81 are actually escape codes,

               skip value minus constant is in the next byte */

            if(!code)

                skip = (*src++) + 64;

            else if(code == 1)

                skip = (*src++) + 320;

            else

                skip = code;

            filled += skip;

            while( filled >= width) {

                filled -= width;

                dst -= stride;

                height--;

                if(height < 0)

                    break;

            }

        } else {

            /* zero code treated as one-pixel skip */

            if(code)

                dst[filled++] = ctable[code & 0x7F];

            else

                filled++;

            if(filled >= width) {

                filled = 0;

                dst -= stride;

                height--;

            }

        }

    }

}
