static void yuvj444p_to_rgb24(AVPicture *dst, AVPicture *src,

                              int width, int height)

{

    uint8_t *y1_ptr, *cb_ptr, *cr_ptr, *d, *d1;

    int w, y, cb, cr, r_add, g_add, b_add;

    uint8_t *cm = cropTbl + MAX_NEG_CROP;

    unsigned int r, g, b;



    d = dst->data[0];

    y1_ptr = src->data[0];

    cb_ptr = src->data[1];

    cr_ptr = src->data[2];

    for(;height > 0; height --) {

        d1 = d;

        for(w = width; w > 0; w--) {

            YUV_TO_RGB1(cb_ptr[0], cr_ptr[0]);



            YUV_TO_RGB2(r, g, b, y1_ptr[0]);

            RGB_OUT(d1, r, g, b);

            d1 += BPP;



            y1_ptr++;

            cb_ptr++;

            cr_ptr++;

        }

        d += dst->linesize[0];

        y1_ptr += src->linesize[0] - width;

        cb_ptr += src->linesize[1] - width;

        cr_ptr += src->linesize[2] - width;

    }

}
