static int h264_slice_header_init(H264Context *h, int reinit)

{

    int nb_slices = (HAVE_THREADS &&

                     h->avctx->active_thread_type & FF_THREAD_SLICE) ?

                    h->avctx->thread_count : 1;

    int i, ret;



    h->avctx->sample_aspect_ratio = h->sps.sar;

    av_assert0(h->avctx->sample_aspect_ratio.den);

    av_pix_fmt_get_chroma_sub_sample(h->avctx->pix_fmt,

                                     &h->chroma_x_shift, &h->chroma_y_shift);



    if (h->sps.timing_info_present_flag) {

        int64_t den = h->sps.time_scale;

        if (h->x264_build < 44U)

            den *= 2;

        av_reduce(&h->avctx->time_base.num, &h->avctx->time_base.den,

                  h->sps.num_units_in_tick, den, 1 << 30);

    }



    if (reinit)

        ff_h264_free_tables(h, 0);

    h->first_field           = 0;

    h->prev_interlaced_frame = 1;



    init_scan_tables(h);

    ret = ff_h264_alloc_tables(h);

    if (ret < 0) {

        av_log(h->avctx, AV_LOG_ERROR, "Could not allocate memory\n");

        return ret;

    }



    if (nb_slices > H264_MAX_THREADS || (nb_slices > h->mb_height && h->mb_height)) {

        int max_slices;

        if (h->mb_height)

            max_slices = FFMIN(H264_MAX_THREADS, h->mb_height);

        else

            max_slices = H264_MAX_THREADS;

        av_log(h->avctx, AV_LOG_WARNING, "too many threads/slices %d,"

               " reducing to %d\n", nb_slices, max_slices);

        nb_slices = max_slices;

    }

    h->slice_context_count = nb_slices;



    if (!HAVE_THREADS || !(h->avctx->active_thread_type & FF_THREAD_SLICE)) {

        ret = ff_h264_context_init(h);

        if (ret < 0) {

            av_log(h->avctx, AV_LOG_ERROR, "context_init() failed.\n");

            return ret;

        }

    } else {

        for (i = 1; i < h->slice_context_count; i++) {

            H264Context *c;

            c                    = h->thread_context[i] = av_mallocz(sizeof(H264Context));

            if (!c)

                return AVERROR(ENOMEM);

            c->avctx             = h->avctx;

            c->dsp               = h->dsp;

            c->vdsp              = h->vdsp;

            c->h264dsp           = h->h264dsp;

            c->h264qpel          = h->h264qpel;

            c->h264chroma        = h->h264chroma;

            c->sps               = h->sps;

            c->pps               = h->pps;

            c->pixel_shift       = h->pixel_shift;

            c->width             = h->width;

            c->height            = h->height;

            c->linesize          = h->linesize;

            c->uvlinesize        = h->uvlinesize;

            c->chroma_x_shift    = h->chroma_x_shift;

            c->chroma_y_shift    = h->chroma_y_shift;

            c->qscale            = h->qscale;

            c->droppable         = h->droppable;

            c->data_partitioning = h->data_partitioning;

            c->low_delay         = h->low_delay;

            c->mb_width          = h->mb_width;

            c->mb_height         = h->mb_height;

            c->mb_stride         = h->mb_stride;

            c->mb_num            = h->mb_num;

            c->flags             = h->flags;

            c->workaround_bugs   = h->workaround_bugs;

            c->pict_type         = h->pict_type;



            init_scan_tables(c);

            clone_tables(c, h, i);

            c->context_initialized = 1;

        }



        for (i = 0; i < h->slice_context_count; i++)

            if ((ret = ff_h264_context_init(h->thread_context[i])) < 0) {

                av_log(h->avctx, AV_LOG_ERROR, "context_init() failed.\n");

                return ret;

            }

    }



    h->context_initialized = 1;



    return 0;

}
