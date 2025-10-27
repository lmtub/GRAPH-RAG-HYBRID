static int alloc_buffer(FrameBuffer **pool, AVCodecContext *s, FrameBuffer **pbuf)

{

    FrameBuffer  *buf = av_mallocz(sizeof(*buf));

    int i, ret;

    const int pixel_size = av_pix_fmt_descriptors[s->pix_fmt].comp[0].step_minus1+1;

    int h_chroma_shift, v_chroma_shift;

    int edge = 32; // XXX should be avcodec_get_edge_width(), but that fails on svq1

    int w = s->width, h = s->height;



    if (!buf)

        return AVERROR(ENOMEM);



    if (!(s->flags & CODEC_FLAG_EMU_EDGE)) {

        w += 2*edge;

        h += 2*edge;

    }



    avcodec_align_dimensions(s, &w, &h);

    if ((ret = av_image_alloc(buf->base, buf->linesize, w, h,

                              s->pix_fmt, 32)) < 0) {

        av_freep(&buf);

        return ret;

    }

    /* XXX this shouldn't be needed, but some tests break without this line

     * those decoders are buggy and need to be fixed.

     * the following tests fail:

     * cdgraphics, ansi, aasc, fraps-v1, qtrle-1bit

     */

    memset(buf->base[0], 128, ret);



    avcodec_get_chroma_sub_sample(s->pix_fmt, &h_chroma_shift, &v_chroma_shift);

    for (i = 0; i < FF_ARRAY_ELEMS(buf->data); i++) {

        const int h_shift = i==0 ? 0 : h_chroma_shift;

        const int v_shift = i==0 ? 0 : v_chroma_shift;

        if (s->flags & CODEC_FLAG_EMU_EDGE)

            buf->data[i] = buf->base[i];

        else

            buf->data[i] = buf->base[i] +

                           FFALIGN((buf->linesize[i]*edge >> v_shift) +

                                   (pixel_size*edge >> h_shift), 32);

    }

    buf->w       = s->width;

    buf->h       = s->height;

    buf->pix_fmt = s->pix_fmt;

    buf->pool    = pool;



    *pbuf = buf;

    return 0;

}
