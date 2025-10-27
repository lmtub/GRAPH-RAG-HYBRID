void filter_level_for_mb(VP8Context *s, VP8Macroblock *mb, VP8FilterStrength *f)

{

    int interior_limit, filter_level;



    if (s->segmentation.enabled) {

        filter_level = s->segmentation.filter_level[mb->segment];

        if (!s->segmentation.absolute_vals)

            filter_level += s->filter.level;

    } else

        filter_level = s->filter.level;



    if (s->lf_delta.enabled) {

        filter_level += s->lf_delta.ref[mb->ref_frame];

        filter_level += s->lf_delta.mode[mb->mode];

    }



    filter_level = av_clip_uintp2(filter_level, 6);



    interior_limit = filter_level;

    if (s->filter.sharpness) {

        interior_limit >>= (s->filter.sharpness + 3) >> 2;

        interior_limit = FFMIN(interior_limit, 9 - s->filter.sharpness);

    }

    interior_limit = FFMAX(interior_limit, 1);



    f->filter_level = filter_level;

    f->inner_limit = interior_limit;

    f->inner_filter = !mb->skip || mb->mode == MODE_I4x4 ||

                      mb->mode == VP8_MVMODE_SPLIT;

}
