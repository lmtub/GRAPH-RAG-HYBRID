static int multiple_resample(ResampleContext *c, AudioData *dst, int dst_size, AudioData *src, int src_size, int *consumed){

    int i, ret= -1;

    int av_unused mm_flags = av_get_cpu_flags();

    int need_emms = c->format == AV_SAMPLE_FMT_S16P && ARCH_X86_32 &&

                    (mm_flags & (AV_CPU_FLAG_MMX2 | AV_CPU_FLAG_SSE2)) == AV_CPU_FLAG_MMX2;

    int64_t max_src_size = (INT64_MAX/2 / c->phase_count) / c->src_incr;



    if (c->compensation_distance)

        dst_size = FFMIN(dst_size, c->compensation_distance);

    src_size = FFMIN(src_size, max_src_size);



    for(i=0; i<dst->ch_count; i++){

        ret= swri_resample(c, dst->ch[i], src->ch[i],

                           consumed, src_size, dst_size, i+1==dst->ch_count);

    }

    if(need_emms)

        emms_c();



    if (c->compensation_distance) {

        c->compensation_distance -= ret;

        if (!c->compensation_distance) {

            c->dst_incr     = c->ideal_dst_incr;

            c->dst_incr_div = c->dst_incr / c->src_incr;

            c->dst_incr_mod = c->dst_incr % c->src_incr;

        }

    }



    return ret;

}
