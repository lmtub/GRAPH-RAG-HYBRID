static int update_context_from_thread(AVCodecContext *dst, AVCodecContext *src, int for_user)

{

    int err = 0;



    if (dst != src) {

        dst->sub_id    = src->sub_id;

        dst->time_base = src->time_base;

        dst->width     = src->width;

        dst->height    = src->height;

        dst->pix_fmt   = src->pix_fmt;



        dst->coded_width  = src->coded_width;

        dst->coded_height = src->coded_height;



        dst->has_b_frames = src->has_b_frames;

        dst->idct_algo    = src->idct_algo;

        dst->slice_count  = src->slice_count;



        dst->bits_per_coded_sample = src->bits_per_coded_sample;

        dst->sample_aspect_ratio   = src->sample_aspect_ratio;

        dst->dtg_active_format     = src->dtg_active_format;



        dst->profile = src->profile;

        dst->level   = src->level;



        dst->bits_per_raw_sample = src->bits_per_raw_sample;

        dst->ticks_per_frame     = src->ticks_per_frame;

        dst->color_primaries     = src->color_primaries;



        dst->color_trc   = src->color_trc;

        dst->colorspace  = src->colorspace;

        dst->color_range = src->color_range;

        dst->chroma_sample_location = src->chroma_sample_location;

    }



    if (for_user) {

        dst->coded_frame   = src->coded_frame;

        dst->has_b_frames += src->thread_count - 1;

    } else {

        if (dst->codec->update_thread_context)

            err = dst->codec->update_thread_context(dst, src);

    }



    return err;

}
