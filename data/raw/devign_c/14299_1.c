int ff_g723_1_scale_vector(int16_t *dst, const int16_t *vector, int length)

{

    int bits, max = 0;

    int i;



    for (i = 0; i < length; i++)

        max |= FFABS(vector[i]);



    bits= 14 - av_log2_16bit(max);

    bits= FFMAX(bits, 0);



    for (i = 0; i < length; i++)

        dst[i] = vector[i] << bits >> 3;



    return bits - 3;

}
