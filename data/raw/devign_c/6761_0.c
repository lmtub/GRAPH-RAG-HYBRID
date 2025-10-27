static int swri_resample(ResampleContext *c,

                         uint8_t *dst, const uint8_t *src, int *consumed,

                         int src_size, int dst_size, int update_ctx)

{

    int fn_idx = c->format - AV_SAMPLE_FMT_S16P;



    if (c->filter_length == 1 && c->phase_shift == 0) {

        int index= c->index;

        int frac= c->frac;

        int64_t index2= (1LL<<32)*c->frac/c->src_incr + (1LL<<32)*index;

        int64_t incr= (1LL<<32) * c->dst_incr / c->src_incr;

        int new_size = (src_size * (int64_t)c->src_incr - frac + c->dst_incr - 1) / c->dst_incr;



        dst_size= FFMIN(dst_size, new_size);

        c->dsp.resample_one[fn_idx](dst, src, dst_size, index2, incr);



        index += dst_size * c->dst_incr_div;

        index += (frac + dst_size * (int64_t)c->dst_incr_mod) / c->src_incr;

        av_assert2(index >= 0);

        *consumed= index;

        if (update_ctx) {

            c->frac   = (frac + dst_size * (int64_t)c->dst_incr_mod) % c->src_incr;

            c->index = 0;

        }

    } else {

        int64_t end_index = (1LL + src_size - c->filter_length) << c->phase_shift;

        int64_t delta_frac = (end_index - c->index) * c->src_incr - c->frac;

        int delta_n = (delta_frac + c->dst_incr - 1) / c->dst_incr;



        dst_size = FFMIN(dst_size, delta_n);

        if (dst_size > 0) {

            if (!c->linear) {

                *consumed = c->dsp.resample_common[fn_idx](c, dst, src, dst_size, update_ctx);

            } else {

                *consumed = c->dsp.resample_linear[fn_idx](c, dst, src, dst_size, update_ctx);

            }

        } else {

            *consumed = 0;

        }

    }



    return dst_size;

}
