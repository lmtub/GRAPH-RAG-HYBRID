static void filter_mb(VP8Context *s, uint8_t *dst[3], VP8Macroblock *mb, int mb_x, int mb_y)

{

    int filter_level, inner_limit, hev_thresh, mbedge_lim, bedge_lim;



    filter_level_for_mb(s, mb, &filter_level, &inner_limit, &hev_thresh);

    if (!filter_level)

        return;



    mbedge_lim = 2*(filter_level+2) + inner_limit;

     bedge_lim = 2* filter_level    + inner_limit;



    if (mb_x) {

        s->vp8dsp.vp8_h_loop_filter16(dst[0], s->linesize,   mbedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter8 (dst[1], s->uvlinesize, mbedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter8 (dst[2], s->uvlinesize, mbedge_lim, inner_limit, hev_thresh);

    }



    if (!mb->skip || mb->mode == MODE_I4x4 || mb->mode == VP8_MVMODE_SPLIT) {

        s->vp8dsp.vp8_h_loop_filter16_inner(dst[0]+ 4, s->linesize,   bedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter16_inner(dst[0]+ 8, s->linesize,   bedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter16_inner(dst[0]+12, s->linesize,   bedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter8_inner (dst[1]+ 4, s->uvlinesize, bedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_h_loop_filter8_inner (dst[2]+ 4, s->uvlinesize, bedge_lim, inner_limit, hev_thresh);

    }



    if (mb_y) {

        s->vp8dsp.vp8_v_loop_filter16(dst[0], s->linesize,   mbedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter8 (dst[1], s->uvlinesize, mbedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter8 (dst[2], s->uvlinesize, mbedge_lim, inner_limit, hev_thresh);

    }



    if (!mb->skip || mb->mode == MODE_I4x4 || mb->mode == VP8_MVMODE_SPLIT) {

        s->vp8dsp.vp8_v_loop_filter16_inner(dst[0]+ 4*s->linesize,   s->linesize,   bedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter16_inner(dst[0]+ 8*s->linesize,   s->linesize,   bedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter16_inner(dst[0]+12*s->linesize,   s->linesize,   bedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter8_inner (dst[1]+ 4*s->uvlinesize, s->uvlinesize, bedge_lim, inner_limit, hev_thresh);

        s->vp8dsp.vp8_v_loop_filter8_inner (dst[2]+ 4*s->uvlinesize, s->uvlinesize, bedge_lim, inner_limit, hev_thresh);

    }

}
