static int alloc_buffers(AVCodecContext *avctx)

{

    CFHDContext *s = avctx->priv_data;

    int i, j, k, ret, planes;



    if ((ret = ff_set_dimensions(avctx, s->coded_width, s->coded_height)) < 0)

        return ret;

    avctx->pix_fmt = s->coded_format;



    avcodec_get_chroma_sub_sample(avctx->pix_fmt, &s->chroma_x_shift, &s->chroma_y_shift);

    planes = av_pix_fmt_count_planes(avctx->pix_fmt);



    for (i = 0; i < planes; i++) {

        int width = i ? avctx->width >> s->chroma_x_shift : avctx->width;

        int height = i ? avctx->height >> s->chroma_y_shift : avctx->height;

        int stride = FFALIGN(width / 8, 8) * 8;

        int w8, h8, w4, h4, w2, h2;

        height = FFALIGN(height / 8, 2) * 8;

        s->plane[i].width = width;

        s->plane[i].height = height;

        s->plane[i].stride = stride;



        w8 = FFALIGN(s->plane[i].width / 8, 8);

        h8 = FFALIGN(s->plane[i].height / 8, 2);

        w4 = w8 * 2;

        h4 = h8 * 2;

        w2 = w4 * 2;

        h2 = h4 * 2;



        s->plane[i].idwt_buf = av_malloc_array(height * stride, sizeof(*s->plane[i].idwt_buf));

        s->plane[i].idwt_tmp = av_malloc_array(height * stride, sizeof(*s->plane[i].idwt_tmp));

        if (!s->plane[i].idwt_buf || !s->plane[i].idwt_tmp) {

            return AVERROR(ENOMEM);

        }



        s->plane[i].subband[0] = s->plane[i].idwt_buf;

        s->plane[i].subband[1] = s->plane[i].idwt_buf + 2 * w8 * h8;

        s->plane[i].subband[2] = s->plane[i].idwt_buf + 1 * w8 * h8;

        s->plane[i].subband[3] = s->plane[i].idwt_buf + 3 * w8 * h8;

        s->plane[i].subband[4] = s->plane[i].idwt_buf + 2 * w4 * h4;

        s->plane[i].subband[5] = s->plane[i].idwt_buf + 1 * w4 * h4;

        s->plane[i].subband[6] = s->plane[i].idwt_buf + 3 * w4 * h4;

        s->plane[i].subband[7] = s->plane[i].idwt_buf + 2 * w2 * h2;

        s->plane[i].subband[8] = s->plane[i].idwt_buf + 1 * w2 * h2;

        s->plane[i].subband[9] = s->plane[i].idwt_buf + 3 * w2 * h2;



        for (j = 0; j < DWT_LEVELS; j++) {

            for(k = 0; k < 4; k++) {

                s->plane[i].band[j][k].a_width  = w8 << j;

                s->plane[i].band[j][k].a_height = h8 << j;

            }

        }



        /* ll2 and ll1 commented out because they are done in-place */

        s->plane[i].l_h[0] = s->plane[i].idwt_tmp;

        s->plane[i].l_h[1] = s->plane[i].idwt_tmp + 2 * w8 * h8;

        //s->plane[i].l_h[2] = ll2;

        s->plane[i].l_h[3] = s->plane[i].idwt_tmp;

        s->plane[i].l_h[4] = s->plane[i].idwt_tmp + 2 * w4 * h4;

        //s->plane[i].l_h[5] = ll1;

        s->plane[i].l_h[6] = s->plane[i].idwt_tmp;

        s->plane[i].l_h[7] = s->plane[i].idwt_tmp + 2 * w2 * h2;

    }



    s->a_height = s->coded_height;

    s->a_width  = s->coded_width;

    s->a_format = s->coded_format;



    return 0;

}
