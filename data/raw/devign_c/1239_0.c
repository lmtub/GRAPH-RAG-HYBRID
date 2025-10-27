RGB_FUNCTIONS(rgb565)



#undef RGB_IN

#undef RGB_OUT

#undef BPP



/* bgr24 handling */



#define RGB_IN(r, g, b, s)\

{\

    b = (s)[0];\

    g = (s)[1];\

    r = (s)[2];\

}



#define RGB_OUT(d, r, g, b)\

{\

    (d)[0] = b;\

    (d)[1] = g;\

    (d)[2] = r;\

}



#define BPP 3



RGB_FUNCTIONS(bgr24)



#undef RGB_IN

#undef RGB_OUT

#undef BPP



/* rgb24 handling */



#define RGB_IN(r, g, b, s)\

{\

    r = (s)[0];\

    g = (s)[1];\

    b = (s)[2];\

}



#define RGB_OUT(d, r, g, b)\

{\

    (d)[0] = r;\

    (d)[1] = g;\

    (d)[2] = b;\

}



#define BPP 3



RGB_FUNCTIONS(rgb24)



static void yuv444p_to_rgb24(AVPicture *dst, AVPicture *src,

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

            YUV_TO_RGB1_CCIR(cb_ptr[0], cr_ptr[0]);



            YUV_TO_RGB2_CCIR(r, g, b, y1_ptr[0]);

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
