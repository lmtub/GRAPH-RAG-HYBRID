static void inline xan_wc3_copy_pixel_run(XanContext *s, 

    int x, int y, int pixel_count, int motion_x, int motion_y)

{

    int stride;

    int line_inc;

    int curframe_index, prevframe_index;

    int curframe_x, prevframe_x;

    int width = s->avctx->width;

    unsigned char *palette_plane, *prev_palette_plane;

    unsigned char *y_plane, *u_plane, *v_plane;

    unsigned char *prev_y_plane, *prev_u_plane, *prev_v_plane;

    unsigned char *rgb_plane, *prev_rgb_plane;

    unsigned short *rgb16_plane, *prev_rgb16_plane;

    unsigned int *rgb32_plane, *prev_rgb32_plane;



    switch (s->avctx->pix_fmt) {



    case PIX_FMT_PAL8:

        palette_plane = s->current_frame.data[0];

        prev_palette_plane = s->last_frame.data[0];

        stride = s->current_frame.linesize[0];

        line_inc = stride - width;

        curframe_index = y * stride + x;

        curframe_x = x;

        prevframe_index = (y + motion_y) * stride + x + motion_x;

        prevframe_x = x + motion_x;

        while(pixel_count--) {



            palette_plane[curframe_index++] = 

                prev_palette_plane[prevframe_index++];



            ADVANCE_CURFRAME_X();

            ADVANCE_PREVFRAME_X();

        }

        break;



    case PIX_FMT_RGB555:

    case PIX_FMT_RGB565:

        rgb16_plane = (unsigned short *)s->current_frame.data[0];

        prev_rgb16_plane = (unsigned short *)s->last_frame.data[0];

        stride = s->current_frame.linesize[0] / 2;

        line_inc = stride - width;

        curframe_index = y * stride + x;

        curframe_x = x;

        prevframe_index = (y + motion_y) * stride + x + motion_x;

        prevframe_x = x + motion_x;

        while(pixel_count--) {



            rgb16_plane[curframe_index++] = 

                prev_rgb16_plane[prevframe_index++];



            ADVANCE_CURFRAME_X();

            ADVANCE_PREVFRAME_X();

        }

        break;



    case PIX_FMT_RGB24:

    case PIX_FMT_BGR24:

        rgb_plane = s->current_frame.data[0];

        prev_rgb_plane = s->last_frame.data[0];

        stride = s->current_frame.linesize[0];

        line_inc = stride - width * 3;

        curframe_index = y * stride + x * 3;

        curframe_x = x;

        prevframe_index = (y + motion_y) * stride + 

            (3 * (x + motion_x));

        prevframe_x = x + motion_x;

        while(pixel_count--) {



            rgb_plane[curframe_index++] = prev_rgb_plane[prevframe_index++];

            rgb_plane[curframe_index++] = prev_rgb_plane[prevframe_index++];

            rgb_plane[curframe_index++] = prev_rgb_plane[prevframe_index++];



            ADVANCE_CURFRAME_X();

            ADVANCE_PREVFRAME_X();

        }

        break;



    case PIX_FMT_RGBA32:

        rgb32_plane = (unsigned int *)s->current_frame.data[0];

        prev_rgb32_plane = (unsigned int *)s->last_frame.data[0];

        stride = s->current_frame.linesize[0] / 4;

        line_inc = stride - width;

        curframe_index = y * stride + x;

        curframe_x = x;

        prevframe_index = (y + motion_y) * stride + x + motion_x;

        prevframe_x = x + motion_x;

        while(pixel_count--) {



            rgb32_plane[curframe_index++] = 

                prev_rgb32_plane[prevframe_index++];



            ADVANCE_CURFRAME_X();

            ADVANCE_PREVFRAME_X();

        }

        break;



    case PIX_FMT_YUV444P:

        y_plane = s->current_frame.data[0];

        u_plane = s->current_frame.data[1];

        v_plane = s->current_frame.data[2];

        prev_y_plane = s->last_frame.data[0];

        prev_u_plane = s->last_frame.data[1];

        prev_v_plane = s->last_frame.data[2];

        stride = s->current_frame.linesize[0];

        line_inc = stride - width;

        curframe_index = y * stride + x;

        curframe_x = x;

        prevframe_index = (y + motion_y) * stride + x + motion_x;

        prevframe_x = x + motion_x;

        while(pixel_count--) {



            y_plane[curframe_index] = prev_y_plane[prevframe_index];

            u_plane[curframe_index] = prev_u_plane[prevframe_index];

            v_plane[curframe_index] = prev_v_plane[prevframe_index];



            curframe_index++;

            ADVANCE_CURFRAME_X();

            prevframe_index++;

            ADVANCE_PREVFRAME_X();

        }

        break;



    default:

        av_log(s->avctx, AV_LOG_ERROR, " Xan WC3: Unhandled colorspace\n");

        break;

    }

}
