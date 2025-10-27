static int config_out_props(AVFilterLink *outlink)

{

    AVFilterContext *ctx = outlink->src;

    AVFilterLink *inlink = outlink->src->inputs[0];

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(outlink->format);

    TInterlaceContext *tinterlace = ctx->priv;

    int i;



    tinterlace->vsub = desc->log2_chroma_h;

    outlink->w = inlink->w;

    outlink->h = tinterlace->mode == MODE_MERGE || tinterlace->mode == MODE_PAD || tinterlace->mode == MODE_MERGEX2?

        inlink->h*2 : inlink->h;

    if (tinterlace->mode == MODE_MERGE || tinterlace->mode == MODE_PAD || tinterlace->mode == MODE_MERGEX2)

        outlink->sample_aspect_ratio = av_mul_q(inlink->sample_aspect_ratio,

                                                av_make_q(2, 1));



    if (tinterlace->mode == MODE_PAD) {

        uint8_t black[4] = { 16, 128, 128, 16 };

        int i, ret;

        if (ff_fmt_is_in(outlink->format, full_scale_yuvj_pix_fmts))

            black[0] = black[3] = 0;

        ret = av_image_alloc(tinterlace->black_data, tinterlace->black_linesize,

                             outlink->w, outlink->h, outlink->format, 16);

        if (ret < 0)

            return ret;



        /* fill black picture with black */

        for (i = 0; i < 4 && tinterlace->black_data[i]; i++) {

            int h = i == 1 || i == 2 ? AV_CEIL_RSHIFT(outlink->h, desc->log2_chroma_h) : outlink->h;

            memset(tinterlace->black_data[i], black[i],

                   tinterlace->black_linesize[i] * h);

        }

    }

    if ((tinterlace->flags & TINTERLACE_FLAG_VLPF

          || tinterlace->flags & TINTERLACE_FLAG_CVLPF)

            && !(tinterlace->mode == MODE_INTERLEAVE_TOP

              || tinterlace->mode == MODE_INTERLEAVE_BOTTOM)) {

        av_log(ctx, AV_LOG_WARNING, "low_pass_filter flags ignored with mode %d\n",

                tinterlace->mode);

        tinterlace->flags &= ~TINTERLACE_FLAG_VLPF;

        tinterlace->flags &= ~TINTERLACE_FLAG_CVLPF;

    }

    tinterlace->preout_time_base = inlink->time_base;

    if (tinterlace->mode == MODE_INTERLACEX2) {

        tinterlace->preout_time_base.den *= 2;

        outlink->frame_rate = av_mul_q(inlink->frame_rate, (AVRational){2,1});

        outlink->time_base  = av_mul_q(inlink->time_base , (AVRational){1,2});

    } else if (tinterlace->mode == MODE_MERGEX2) {

        outlink->frame_rate = inlink->frame_rate;

        outlink->time_base  = inlink->time_base;

    } else if (tinterlace->mode != MODE_PAD) {

        outlink->frame_rate = av_mul_q(inlink->frame_rate, (AVRational){1,2});

        outlink->time_base  = av_mul_q(inlink->time_base , (AVRational){2,1});

    }



    for (i = 0; i<FF_ARRAY_ELEMS(standard_tbs); i++){

        if (!av_cmp_q(standard_tbs[i], outlink->time_base))

            break;

    }

    if (i == FF_ARRAY_ELEMS(standard_tbs) ||

        (tinterlace->flags & TINTERLACE_FLAG_EXACT_TB))

        outlink->time_base = tinterlace->preout_time_base;



    if (tinterlace->flags & TINTERLACE_FLAG_CVLPF) {

        tinterlace->lowpass_line = lowpass_line_complex_c;

        if (ARCH_X86)

            ff_tinterlace_init_x86(tinterlace);

    } else if (tinterlace->flags & TINTERLACE_FLAG_VLPF) {

        tinterlace->lowpass_line = lowpass_line_c;

        if (ARCH_X86)

            ff_tinterlace_init_x86(tinterlace);

    }



    av_log(ctx, AV_LOG_VERBOSE, "mode:%d filter:%s h:%d -> h:%d\n", tinterlace->mode,

           (tinterlace->flags & TINTERLACE_FLAG_CVLPF) ? "complex" :

           (tinterlace->flags & TINTERLACE_FLAG_VLPF) ? "linear" : "off",

           inlink->h, outlink->h);



    return 0;

}
