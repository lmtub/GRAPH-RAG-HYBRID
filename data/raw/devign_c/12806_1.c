static int h264_slice_header_init(H264Context *h, int reinit)

{

    MpegEncContext *const s  = &h->s;

    int i, ret;



    if(   FFALIGN(s->avctx->width , 16                                 ) == s->width

        && FFALIGN(s->avctx->height, 16*(2 - h->sps.frame_mbs_only_flag)) == s->height

        && !h->sps.crop_right && !h->sps.crop_bottom

        && (s->avctx->width != s->width || s->avctx->height && s->height)

    ) {

        av_log(h->s.avctx, AV_LOG_DEBUG, "Using externally provided dimensions\n");

        s->avctx->coded_width  = s->width;

        s->avctx->coded_height = s->height;

    } else{

        avcodec_set_dimensions(s->avctx, s->width, s->height);

        s->avctx->width  -= (2>>CHROMA444)*FFMIN(h->sps.crop_right, (8<<CHROMA444)-1);

        s->avctx->height -= (1<<s->chroma_y_shift)*FFMIN(h->sps.crop_bottom, (16>>s->chroma_y_shift)-1) * (2 - h->sps.frame_mbs_only_flag);

    }



    s->avctx->sample_aspect_ratio = h->sps.sar;

    av_assert0(s->avctx->sample_aspect_ratio.den);



    if (h->sps.timing_info_present_flag) {

        int64_t den = h->sps.time_scale;

        if (h->x264_build < 44U)

            den *= 2;

        av_reduce(&s->avctx->time_base.num, &s->avctx->time_base.den,

                  h->sps.num_units_in_tick, den, 1 << 30);

    }



    s->avctx->hwaccel = ff_find_hwaccel(s->avctx->codec->id, s->avctx->pix_fmt);



    if (reinit) {

        free_tables(h, 0);

        if ((ret = ff_MPV_common_frame_size_change(s)) < 0) {

            av_log(h->s.avctx, AV_LOG_ERROR, "ff_MPV_common_frame_size_change() failed.\n");

            return ret;

        }

    } else {

        if ((ret = ff_MPV_common_init(s) < 0)) {

            av_log(h->s.avctx, AV_LOG_ERROR, "ff_MPV_common_init() failed.\n");

            return ret;

        }

    }

    s->first_field = 0;

    h->prev_interlaced_frame = 1;



    init_scan_tables(h);

    if (ff_h264_alloc_tables(h) < 0) {

        av_log(h->s.avctx, AV_LOG_ERROR,

               "Could not allocate memory for h264\n");

        return AVERROR(ENOMEM);

    }



    if (!HAVE_THREADS || !(s->avctx->active_thread_type & FF_THREAD_SLICE)) {

        if (context_init(h) < 0) {

            av_log(h->s.avctx, AV_LOG_ERROR, "context_init() failed.\n");

            return -1;

        }

    } else {

        for (i = 1; i < s->slice_context_count; i++) {

            H264Context *c;

            c = h->thread_context[i] = av_malloc(sizeof(H264Context));

            memcpy(c, h->s.thread_context[i], sizeof(MpegEncContext));

            memset(&c->s + 1, 0, sizeof(H264Context) - sizeof(MpegEncContext));

            c->h264dsp     = h->h264dsp;

            c->sps         = h->sps;

            c->pps         = h->pps;

            c->pixel_shift = h->pixel_shift;

            c->cur_chroma_format_idc = h->cur_chroma_format_idc;

            init_scan_tables(c);

            clone_tables(c, h, i);

        }



        for (i = 0; i < s->slice_context_count; i++)

            if (context_init(h->thread_context[i]) < 0) {

                av_log(h->s.avctx, AV_LOG_ERROR, "context_init() failed.\n");

                return -1;

            }

    }



    return 0;

}
