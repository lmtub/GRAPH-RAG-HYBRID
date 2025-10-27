static int lag_decode_frame(AVCodecContext *avctx,

                            void *data, int *got_frame, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    unsigned int buf_size = avpkt->size;

    LagarithContext *l = avctx->priv_data;

    ThreadFrame frame = { .f = data };

    AVFrame *const p  = data;

    uint8_t frametype = 0;

    uint32_t offset_gu = 0, offset_bv = 0, offset_ry = 9;

    uint32_t offs[4];

    uint8_t *srcs[4], *dst;

    int i, j, planes = 3;

    int ret;



    p->key_frame = 1;



    frametype = buf[0];



    offset_gu = AV_RL32(buf + 1);

    offset_bv = AV_RL32(buf + 5);



    switch (frametype) {

    case FRAME_SOLID_RGBA:

        avctx->pix_fmt = AV_PIX_FMT_RGB32;

    case FRAME_SOLID_GRAY:

        if (frametype == FRAME_SOLID_GRAY)

            if (avctx->bits_per_coded_sample == 24) {

                avctx->pix_fmt = AV_PIX_FMT_RGB24;

            } else {

                avctx->pix_fmt = AV_PIX_FMT_0RGB32;

                planes = 4;

            }



        if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)

            return ret;



        dst = p->data[0];

        if (frametype == FRAME_SOLID_RGBA) {

        for (j = 0; j < avctx->height; j++) {

            for (i = 0; i < avctx->width; i++)

                AV_WN32(dst + i * 4, offset_gu);

            dst += p->linesize[0];

        }

        } else {

            for (j = 0; j < avctx->height; j++) {

                memset(dst, buf[1], avctx->width * planes);

                dst += p->linesize[0];

            }

        }

        break;

    case FRAME_SOLID_COLOR:

        if (avctx->bits_per_coded_sample == 24) {

            avctx->pix_fmt = AV_PIX_FMT_RGB24;

        } else {

            avctx->pix_fmt = AV_PIX_FMT_RGB32;

            offset_gu |= 0xFFU << 24;

        }



        if ((ret = ff_thread_get_buffer(avctx, &frame,0)) < 0)

            return ret;



        dst = p->data[0];

        for (j = 0; j < avctx->height; j++) {

            for (i = 0; i < avctx->width; i++)

                if (avctx->bits_per_coded_sample == 24) {

                    AV_WB24(dst + i * 3, offset_gu);

                } else {

                    AV_WN32(dst + i * 4, offset_gu);

                }

            dst += p->linesize[0];

        }

        break;

    case FRAME_ARITH_RGBA:

        avctx->pix_fmt = AV_PIX_FMT_RGB32;

        planes = 4;

        offset_ry += 4;

        offs[3] = AV_RL32(buf + 9);

    case FRAME_ARITH_RGB24:

    case FRAME_U_RGB24:

        if (frametype == FRAME_ARITH_RGB24 || frametype == FRAME_U_RGB24)

            avctx->pix_fmt = AV_PIX_FMT_RGB24;



        if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)

            return ret;



        offs[0] = offset_bv;

        offs[1] = offset_gu;

        offs[2] = offset_ry;



        l->rgb_stride = FFALIGN(avctx->width, 16);

        av_fast_malloc(&l->rgb_planes, &l->rgb_planes_allocated,

                       l->rgb_stride * avctx->height * planes + 1);

        if (!l->rgb_planes) {

            av_log(avctx, AV_LOG_ERROR, "cannot allocate temporary buffer\n");

            return AVERROR(ENOMEM);

        }

        for (i = 0; i < planes; i++)

            srcs[i] = l->rgb_planes + (i + 1) * l->rgb_stride * avctx->height - l->rgb_stride;

        for (i = 0; i < planes; i++)

            if (buf_size <= offs[i]) {

                av_log(avctx, AV_LOG_ERROR,

                        "Invalid frame offsets\n");

                return AVERROR_INVALIDDATA;

            }



        for (i = 0; i < planes; i++)

            lag_decode_arith_plane(l, srcs[i],

                                   avctx->width, avctx->height,

                                   -l->rgb_stride, buf + offs[i],

                                   buf_size - offs[i]);

        dst = p->data[0];

        for (i = 0; i < planes; i++)

            srcs[i] = l->rgb_planes + i * l->rgb_stride * avctx->height;

        for (j = 0; j < avctx->height; j++) {

            for (i = 0; i < avctx->width; i++) {

                uint8_t r, g, b, a;

                r = srcs[0][i];

                g = srcs[1][i];

                b = srcs[2][i];

                r += g;

                b += g;

                if (frametype == FRAME_ARITH_RGBA) {

                    a = srcs[3][i];

                    AV_WN32(dst + i * 4, MKBETAG(a, r, g, b));

                } else {

                    dst[i * 3 + 0] = r;

                    dst[i * 3 + 1] = g;

                    dst[i * 3 + 2] = b;

                }

            }

            dst += p->linesize[0];

            for (i = 0; i < planes; i++)

                srcs[i] += l->rgb_stride;

        }

        break;

    case FRAME_ARITH_YUY2:

        avctx->pix_fmt = AV_PIX_FMT_YUV422P;



        if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)

            return ret;



        if (offset_ry >= buf_size ||

            offset_gu >= buf_size ||

            offset_bv >= buf_size) {

            av_log(avctx, AV_LOG_ERROR,

                   "Invalid frame offsets\n");

            return AVERROR_INVALIDDATA;

        }



        lag_decode_arith_plane(l, p->data[0], avctx->width, avctx->height,

                               p->linesize[0], buf + offset_ry,

                               buf_size - offset_ry);

        lag_decode_arith_plane(l, p->data[1], avctx->width / 2,

                               avctx->height, p->linesize[1],

                               buf + offset_gu, buf_size - offset_gu);

        lag_decode_arith_plane(l, p->data[2], avctx->width / 2,

                               avctx->height, p->linesize[2],

                               buf + offset_bv, buf_size - offset_bv);

        break;

    case FRAME_ARITH_YV12:

        avctx->pix_fmt = AV_PIX_FMT_YUV420P;



        if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)

            return ret;

        if (buf_size <= offset_ry || buf_size <= offset_gu || buf_size <= offset_bv) {

            return AVERROR_INVALIDDATA;

        }



        if (offset_ry >= buf_size ||

            offset_gu >= buf_size ||

            offset_bv >= buf_size) {

            av_log(avctx, AV_LOG_ERROR,

                   "Invalid frame offsets\n");

            return AVERROR_INVALIDDATA;

        }



        lag_decode_arith_plane(l, p->data[0], avctx->width, avctx->height,

                               p->linesize[0], buf + offset_ry,

                               buf_size - offset_ry);

        lag_decode_arith_plane(l, p->data[2], avctx->width / 2,

                               avctx->height / 2, p->linesize[2],

                               buf + offset_gu, buf_size - offset_gu);

        lag_decode_arith_plane(l, p->data[1], avctx->width / 2,

                               avctx->height / 2, p->linesize[1],

                               buf + offset_bv, buf_size - offset_bv);

        break;

    default:

        av_log(avctx, AV_LOG_ERROR,

               "Unsupported Lagarith frame type: %#"PRIx8"\n", frametype);

        return AVERROR_PATCHWELCOME;

    }



    *got_frame = 1;



    return buf_size;

}
