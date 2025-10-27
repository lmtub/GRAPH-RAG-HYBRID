int img_convert(AVPicture *dst, int dst_pix_fmt,

                AVPicture *src, int src_pix_fmt, 

                int src_width, int src_height)

{

    static int inited;

    int i, ret, dst_width, dst_height, int_pix_fmt;

    PixFmtInfo *src_pix, *dst_pix;

    ConvertEntry *ce;

    AVPicture tmp1, *tmp = &tmp1;



    if (src_pix_fmt < 0 || src_pix_fmt >= PIX_FMT_NB ||

        dst_pix_fmt < 0 || dst_pix_fmt >= PIX_FMT_NB)

        return -1;

    if (src_width <= 0 || src_height <= 0)

        return 0;



    if (!inited) {

        inited = 1;

        img_convert_init();

    }



    dst_width = src_width;

    dst_height = src_height;



    dst_pix = &pix_fmt_info[dst_pix_fmt];

    src_pix = &pix_fmt_info[src_pix_fmt];

    if (src_pix_fmt == dst_pix_fmt) {

        /* XXX: incorrect */

        /* same format: just copy */

        for(i = 0; i < dst_pix->nb_components; i++) {

            int w, h;

            w = dst_width;

            h = dst_height;

            if (is_yuv_planar(dst_pix) && (i == 1 || i == 2)) {

                w >>= dst_pix->x_chroma_shift;

                h >>= dst_pix->y_chroma_shift;

            }

            img_copy(dst->data[i], dst->linesize[i],

                     src->data[i], src->linesize[i],

                     w, h);

        }

        return 0;

    }



    ce = &convert_table[src_pix_fmt][dst_pix_fmt];

    if (ce->convert) {

        /* specific convertion routine */

        ce->convert(dst, src, dst_width, dst_height);

        return 0;

    }



    /* gray to YUV */

    if (is_yuv_planar(dst_pix) &&

        src_pix_fmt == PIX_FMT_GRAY8) {

        int w, h, y;

        uint8_t *d;



        if (dst_pix->color_type == FF_COLOR_YUV_JPEG) {

            img_copy(dst->data[0], dst->linesize[0],

                     src->data[0], src->linesize[0],

                     dst_width, dst_height);

        } else {

            img_apply_table(dst->data[0], dst->linesize[0],

                            src->data[0], src->linesize[0],

                            dst_width, dst_height,

                            y_jpeg_to_ccir);

        }

        /* fill U and V with 128 */

        w = dst_width;

        h = dst_height;

        w >>= dst_pix->x_chroma_shift;

        h >>= dst_pix->y_chroma_shift;

        for(i = 1; i <= 2; i++) {

            d = dst->data[i];

            for(y = 0; y< h; y++) {

                memset(d, 128, w);

                d += dst->linesize[i];

            }

        }

        return 0;

    }



    /* YUV to gray */

    if (is_yuv_planar(src_pix) && 

        dst_pix_fmt == PIX_FMT_GRAY8) {

        if (src_pix->color_type == FF_COLOR_YUV_JPEG) {

            img_copy(dst->data[0], dst->linesize[0],

                     src->data[0], src->linesize[0],

                     dst_width, dst_height);

        } else {

            img_apply_table(dst->data[0], dst->linesize[0],

                            src->data[0], src->linesize[0],

                            dst_width, dst_height,

                            y_ccir_to_jpeg);

        }

        return 0;

    }



    /* YUV to YUV planar */

    if (is_yuv_planar(dst_pix) && is_yuv_planar(src_pix)) {

        int x_shift, y_shift, w, h;

        void (*resize_func)(uint8_t *dst, int dst_wrap, 

                            uint8_t *src, int src_wrap,

                            int width, int height);



        /* compute chroma size of the smallest dimensions */

        w = dst_width;

        h = dst_height;

        if (dst_pix->x_chroma_shift >= src_pix->x_chroma_shift)

            w >>= dst_pix->x_chroma_shift;

        else

            w >>= src_pix->x_chroma_shift;

        if (dst_pix->y_chroma_shift >= src_pix->y_chroma_shift)

            h >>= dst_pix->y_chroma_shift;

        else

            h >>= src_pix->y_chroma_shift;



        x_shift = (dst_pix->x_chroma_shift - src_pix->x_chroma_shift);

        y_shift = (dst_pix->y_chroma_shift - src_pix->y_chroma_shift);



        if (x_shift == 0 && y_shift == 0) {

            resize_func = img_copy;

        } else if (x_shift == 0 && y_shift == 1) {

            resize_func = shrink2;

        } else if (x_shift == 1 && y_shift == 1) {

            resize_func = shrink22;

        } else if (x_shift == -1 && y_shift == -1) {

            resize_func = grow22;

        } else if (x_shift == -1 && y_shift == 1) {

            resize_func = conv411;

        } else {

            /* currently not handled */

            return -1;

        }



        img_copy(dst->data[0], dst->linesize[0],

                 src->data[0], src->linesize[0],

                 dst_width, dst_height);



        for(i = 1;i <= 2; i++)

            resize_func(dst->data[i], dst->linesize[i],

                        src->data[i], src->linesize[i],

                        dst_width>>dst_pix->x_chroma_shift, dst_height>>dst_pix->y_chroma_shift);

        /* if yuv color space conversion is needed, we do it here on

           the destination image */

        if (dst_pix->color_type != src_pix->color_type) {

            const uint8_t *y_table, *c_table;

            if (dst_pix->color_type == FF_COLOR_YUV) {

                y_table = y_jpeg_to_ccir;

                c_table = c_jpeg_to_ccir;

            } else {

                y_table = y_ccir_to_jpeg;

                c_table = c_ccir_to_jpeg;

            }

            img_apply_table(dst->data[0], dst->linesize[0],

                            dst->data[0], dst->linesize[0],

                            dst_width, dst_height,

                            y_table);



            for(i = 1;i <= 2; i++)

                img_apply_table(dst->data[i], dst->linesize[i],

                                dst->data[i], dst->linesize[i],

                                dst_width>>dst_pix->x_chroma_shift, 

                                dst_height>>dst_pix->y_chroma_shift,

                                c_table);

        }

        return 0;

    }



    /* try to use an intermediate format */

    if (src_pix_fmt == PIX_FMT_YUV422 ||

        dst_pix_fmt == PIX_FMT_YUV422) {

        /* specific case: convert to YUV422P first */

        int_pix_fmt = PIX_FMT_YUV422P;

    } else if ((src_pix->color_type == FF_COLOR_GRAY &&

                src_pix_fmt != PIX_FMT_GRAY8) || 

               (dst_pix->color_type == FF_COLOR_GRAY &&

                dst_pix_fmt != PIX_FMT_GRAY8)) {

        /* gray8 is the normalized format */

        int_pix_fmt = PIX_FMT_GRAY8;

    } else if ((is_yuv_planar(src_pix) && 

                src_pix_fmt != PIX_FMT_YUV444P &&

                src_pix_fmt != PIX_FMT_YUVJ444P)) {

        /* yuv444 is the normalized format */

        if (src_pix->color_type == FF_COLOR_YUV_JPEG)

            int_pix_fmt = PIX_FMT_YUVJ444P;

        else

            int_pix_fmt = PIX_FMT_YUV444P;

    } else if ((is_yuv_planar(dst_pix) && 

                dst_pix_fmt != PIX_FMT_YUV444P &&

                dst_pix_fmt != PIX_FMT_YUVJ444P)) {

        /* yuv444 is the normalized format */

        if (dst_pix->color_type == FF_COLOR_YUV_JPEG)

            int_pix_fmt = PIX_FMT_YUVJ444P;

        else

            int_pix_fmt = PIX_FMT_YUV444P;

    } else {

        /* the two formats are rgb or gray8 or yuv[j]444p */

        if (src_pix->is_alpha && dst_pix->is_alpha)

            int_pix_fmt = PIX_FMT_RGBA32;

        else

            int_pix_fmt = PIX_FMT_RGB24;

    }

    if (avpicture_alloc(tmp, int_pix_fmt, dst_width, dst_height) < 0)

        return -1;

    ret = -1;

    if (img_convert(tmp, int_pix_fmt,

                    src, src_pix_fmt, src_width, src_height) < 0)

        goto fail1;

    if (img_convert(dst, dst_pix_fmt,

                    tmp, int_pix_fmt, dst_width, dst_height) < 0)

        goto fail1;

    ret = 0;

 fail1:

    avpicture_free(tmp);

    return ret;

}
