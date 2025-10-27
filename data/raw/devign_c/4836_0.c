static inline void conv_to_int32(int32_t *loc, float *samples, int num, float norm)

{

    int i;

    for (i = 0; i < num; i++)

        loc[i] = ceilf((samples[i]/norm)*INT32_MAX);

}
