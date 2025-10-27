static av_always_inline int rv40_loop_filter_strength(uint8_t *src,

                                                      int step, int stride,

                                                      int beta, int beta2,

                                                      int edge,

                                                      int *p1, int *q1)

{

    int sum_p1p0 = 0, sum_q1q0 = 0, sum_p1p2 = 0, sum_q1q2 = 0;

    int strong0 = 0, strong1 = 0;

    uint8_t *ptr;

    int i;



    for (i = 0, ptr = src; i < 4; i++, ptr += stride) {

        sum_p1p0 += ptr[-2*step] - ptr[-1*step];

        sum_q1q0 += ptr[ 1*step] - ptr[ 0*step];

    }



    *p1 = FFABS(sum_p1p0) < (beta << 2);

    *q1 = FFABS(sum_q1q0) < (beta << 2);



    if(!*p1 && !*q1)

        return 0;



    if (!edge)

        return 0;



    for (i = 0, ptr = src; i < 4; i++, ptr += stride) {

        sum_p1p2 += ptr[-2*step] - ptr[-3*step];

        sum_q1q2 += ptr[ 1*step] - ptr[ 2*step];

    }



    strong0 = *p1 && (FFABS(sum_p1p2) < beta2);

    strong1 = *q1 && (FFABS(sum_q1q2) < beta2);



    return strong0 && strong1;

}
