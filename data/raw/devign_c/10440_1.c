static int verify_md5(HEVCContext *s, AVFrame *frame)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frame->format);

    int pixel_shift = desc->comp[0].depth_minus1 > 7;

    int i, j;



    if (!desc)

        return AVERROR(EINVAL);



    av_log(s->avctx, AV_LOG_DEBUG, "Verifying checksum for frame with POC %d: ",

           s->poc);



    /* the checksums are LE, so we have to byteswap for >8bpp formats

     * on BE arches */

#if HAVE_BIGENDIAN

    if (pixel_shift && !s->checksum_buf) {

        av_fast_malloc(&s->checksum_buf, &s->checksum_buf_size,

                       FFMAX3(frame->linesize[0], frame->linesize[1],

                              frame->linesize[2]));

        if (!s->checksum_buf)

            return AVERROR(ENOMEM);

    }

#endif



    for (i = 0; frame->data[i]; i++) {

        int width  = s->avctx->coded_width;

        int height = s->avctx->coded_height;

        int w = (i == 1 || i == 2) ? (width  >> desc->log2_chroma_w) : width;

        int h = (i == 1 || i == 2) ? (height >> desc->log2_chroma_h) : height;

        uint8_t md5[16];



        av_md5_init(s->md5_ctx);

        for (j = 0; j < h; j++) {

            const uint8_t *src = frame->data[i] + j * frame->linesize[i];

#if HAVE_BIGENDIAN

            if (pixel_shift) {

                s->dsp.bswap16_buf((uint16_t*)s->checksum_buf,

                                   (const uint16_t*)src, w);

                src = s->checksum_buf;

            }

#endif

            av_md5_update(s->md5_ctx, src, w << pixel_shift);

        }

        av_md5_final(s->md5_ctx, md5);



        if (!memcmp(md5, s->md5[i], 16)) {

            av_log   (s->avctx, AV_LOG_DEBUG, "plane %d - correct ", i);

            print_md5(s->avctx, AV_LOG_DEBUG, md5);

            av_log   (s->avctx, AV_LOG_DEBUG, "; ");

        } else {

            av_log   (s->avctx, AV_LOG_ERROR, "mismatching checksum of plane %d - ", i);

            print_md5(s->avctx, AV_LOG_ERROR, md5);

            av_log   (s->avctx, AV_LOG_ERROR, " != ");

            print_md5(s->avctx, AV_LOG_ERROR, s->md5[i]);

            av_log   (s->avctx, AV_LOG_ERROR, "\n");

            return AVERROR_INVALIDDATA;

        }

    }



    av_log(s->avctx, AV_LOG_DEBUG, "\n");



    return 0;

}
