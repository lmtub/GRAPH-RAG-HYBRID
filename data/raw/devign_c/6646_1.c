static av_always_inline void rv40_weak_loop_filter(uint8_t *src,

                                                   const int step,

                                                   const int stride,

                                                   const int filter_p1,

                                                   const int filter_q1,

                                                   const int alpha,

                                                   const int beta,

                                                   const int lim_p0q0,

                                                   const int lim_q1,

                                                   const int lim_p1)

{

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;

    int i, t, u, diff;



    for (i = 0; i < 4; i++, src += stride) {

        int diff_p1p0 = src[-2*step] - src[-1*step];

        int diff_q1q0 = src[ 1*step] - src[ 0*step];

        int diff_p1p2 = src[-2*step] - src[-3*step];

        int diff_q1q2 = src[ 1*step] - src[ 2*step];



        t = src[0*step] - src[-1*step];

        if (!t)

            continue;



        u = (alpha * FFABS(t)) >> 7;

        if (u > 3 - (filter_p1 && filter_q1))

            continue;



        t <<= 2;

        if (filter_p1 && filter_q1)

            t += src[-2*step] - src[1*step];



        diff = CLIP_SYMM((t + 4) >> 3, lim_p0q0);

        src[-1*step] = cm[src[-1*step] + diff];

        src[ 0*step] = cm[src[ 0*step] - diff];



        if (filter_p1 && FFABS(diff_p1p2) <= beta) {

            t = (diff_p1p0 + diff_p1p2 - diff) >> 1;

            src[-2*step] = cm[src[-2*step] - CLIP_SYMM(t, lim_p1)];

        }



        if (filter_q1 && FFABS(diff_q1q2) <= beta) {

            t = (diff_q1q0 + diff_q1q2 + diff) >> 1;

            src[ 1*step] = cm[src[ 1*step] - CLIP_SYMM(t, lim_q1)];

        }

    }

}
