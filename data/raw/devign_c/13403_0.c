static inline int round_sample(int64_t *sum)

{

    int sum1;

    sum1 = (int)((*sum) >> OUT_SHIFT);

    *sum &= (1<<OUT_SHIFT)-1;

    return av_clip(sum1, OUT_MIN, OUT_MAX);

}
