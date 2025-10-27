static int filter_frame(AVFilterLink *link, AVFrame *in)

{

    AVFilterContext *ctx = link->dst;

    AVFilterLink *outlink = ctx->outputs[0];

    ColorSpaceContext *s = ctx->priv;

    // FIXME if yuv2yuv_passthrough, don't get a new buffer but use the

    // input one if it is writable *OR* the actual literal values of in_*

    // and out_* are identical (not just their respective properties)

    AVFrame *out = ff_get_video_buffer(outlink, outlink->w, outlink->h);

    int res;

    ptrdiff_t rgb_stride = FFALIGN(in->width * sizeof(int16_t), 32);

    unsigned rgb_sz = rgb_stride * in->height;

    struct ThreadData td;



    if (!out) {

        av_frame_free(&in);

        return AVERROR(ENOMEM);

    }

    av_frame_copy_props(out, in);



    out->color_primaries = s->user_prm == AVCOL_PRI_UNSPECIFIED ?

                           default_prm[FFMIN(s->user_all, CS_NB)] : s->user_prm;

    if (s->user_trc == AVCOL_TRC_UNSPECIFIED) {

        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(out->format);



        out->color_trc   = default_trc[FFMIN(s->user_all, CS_NB)];

        if (out->color_trc == AVCOL_TRC_BT2020_10 && desc && desc->comp[0].depth >= 12)

            out->color_trc = AVCOL_TRC_BT2020_12;

    } else {

        out->color_trc   = s->user_trc;

    }

    out->colorspace      = s->user_csp == AVCOL_SPC_UNSPECIFIED ?

                           default_csp[FFMIN(s->user_all, CS_NB)] : s->user_csp;

    out->color_range     = s->user_rng == AVCOL_RANGE_UNSPECIFIED ?

                           in->color_range : s->user_rng;

    if (rgb_sz != s->rgb_sz) {

        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(out->format);

        int uvw = in->width >> desc->log2_chroma_w;



        av_freep(&s->rgb[0]);

        av_freep(&s->rgb[1]);

        av_freep(&s->rgb[2]);

        s->rgb_sz = 0;

        av_freep(&s->dither_scratch_base[0][0]);

        av_freep(&s->dither_scratch_base[0][1]);

        av_freep(&s->dither_scratch_base[1][0]);

        av_freep(&s->dither_scratch_base[1][1]);

        av_freep(&s->dither_scratch_base[2][0]);

        av_freep(&s->dither_scratch_base[2][1]);



        s->rgb[0] = av_malloc(rgb_sz);

        s->rgb[1] = av_malloc(rgb_sz);

        s->rgb[2] = av_malloc(rgb_sz);

        s->dither_scratch_base[0][0] =

            av_malloc(sizeof(*s->dither_scratch_base[0][0]) * (in->width + 4));

        s->dither_scratch_base[0][1] =

            av_malloc(sizeof(*s->dither_scratch_base[0][1]) * (in->width + 4));

        s->dither_scratch_base[1][0] =

            av_malloc(sizeof(*s->dither_scratch_base[1][0]) * (uvw + 4));

        s->dither_scratch_base[1][1] =

            av_malloc(sizeof(*s->dither_scratch_base[1][1]) * (uvw + 4));

        s->dither_scratch_base[2][0] =

            av_malloc(sizeof(*s->dither_scratch_base[2][0]) * (uvw + 4));

        s->dither_scratch_base[2][1] =

            av_malloc(sizeof(*s->dither_scratch_base[2][1]) * (uvw + 4));

        s->dither_scratch[0][0] = &s->dither_scratch_base[0][0][1];

        s->dither_scratch[0][1] = &s->dither_scratch_base[0][1][1];

        s->dither_scratch[1][0] = &s->dither_scratch_base[1][0][1];

        s->dither_scratch[1][1] = &s->dither_scratch_base[1][1][1];

        s->dither_scratch[2][0] = &s->dither_scratch_base[2][0][1];

        s->dither_scratch[2][1] = &s->dither_scratch_base[2][1][1];

        if (!s->rgb[0] || !s->rgb[1] || !s->rgb[2] ||

            !s->dither_scratch_base[0][0] || !s->dither_scratch_base[0][1] ||

            !s->dither_scratch_base[1][0] || !s->dither_scratch_base[1][1] ||

            !s->dither_scratch_base[2][0] || !s->dither_scratch_base[2][1]) {

            uninit(ctx);

            return AVERROR(ENOMEM);

        }

        s->rgb_sz = rgb_sz;

    }

    res = create_filtergraph(ctx, in, out);

    if (res < 0)

        return res;

    s->rgb_stride = rgb_stride / sizeof(int16_t);

    td.in = in;

    td.out = out;

    td.in_linesize[0] = in->linesize[0];

    td.in_linesize[1] = in->linesize[1];

    td.in_linesize[2] = in->linesize[2];

    td.out_linesize[0] = out->linesize[0];

    td.out_linesize[1] = out->linesize[1];

    td.out_linesize[2] = out->linesize[2];

    td.in_ss_h = av_pix_fmt_desc_get(in->format)->log2_chroma_h;

    td.out_ss_h = av_pix_fmt_desc_get(out->format)->log2_chroma_h;

    if (s->yuv2yuv_passthrough) {

        av_frame_copy(out, in);

    } else {

        ctx->internal->execute(ctx, convert, &td, NULL,

                               FFMIN((in->height + 1) >> 1, ctx->graph->nb_threads));

    }

    av_frame_free(&in);



    return ff_filter_frame(outlink, out);

}
