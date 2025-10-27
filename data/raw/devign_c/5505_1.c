void filter_mb(VP8Context *s, uint8_t *dst[3], VP8FilterStrength *f,

               int mb_x, int mb_y)

{

    int mbedge_lim, bedge_lim, hev_thresh;

    int filter_level = f->filter_level;

    int inner_limit = f->inner_limit;

    int inner_filter = f->inner_filter;

    int linesize = s->linesize;

    int uvlinesize = s->uvlinesize;

    static const uint8_t hev_thresh_lut[2][64] = {

        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,

          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,

          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,

          3, 3, 3, 3 },

        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,

          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,

          2, 2, 2, 2 }

    };



    if (!filter_level)

        return;



    bedge_lim  = 2 * filter_level + inner_limit;

    mbedge_lim = bedge_lim + 4;



    hev_thresh = hev_thresh_lut[s->keyframe][filter_level];



    if (mb_x) {

        s->vp8dsp.vp8_h_loop_filter16y(dst[0], linesize,

                                       mbedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter8uv(dst[1], dst[2], uvlinesize,

                                       mbedge_lim, inner_limit, hev_thresh);

    }



    if (inner_filter) {

        s->vp8dsp.vp8_h_loop_filter16y_inner(dst[0] +  4, linesize, bedge_lim,

                                             inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter16y_inner(dst[0] +  8, linesize, bedge_lim,

                                             inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter16y_inner(dst[0] + 12, linesize, bedge_lim,

                                             inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter8uv_inner(dst[1] +  4, dst[2] + 4,

                                             uvlinesize, bedge_lim,

                                             inner_limit, hev_thresh);

    }



    if (mb_y) {

        s->vp8dsp.vp8_v_loop_filter16y(dst[0], linesize,

                                       mbedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter8uv(dst[1], dst[2], uvlinesize,

                                       mbedge_lim, inner_limit, hev_thresh);

    }



    if (inner_filter) {

        s->vp8dsp.vp8_v_loop_filter16y_inner(dst[0] +  4 * linesize,

                                             linesize, bedge_lim,

                                             inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter16y_inner(dst[0] +  8 * linesize,

                                             linesize, bedge_lim,

                                             inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter16y_inner(dst[0] + 12 * linesize,

                                             linesize, bedge_lim,

                                             inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter8uv_inner(dst[1] +  4 * uvlinesize,

                                             dst[2] +  4 * uvlinesize,

                                             uvlinesize, bedge_lim,

                                             inner_limit, hev_thresh);

    }

}
