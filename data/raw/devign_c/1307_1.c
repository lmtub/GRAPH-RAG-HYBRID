FFTContext *av_fft_init(int nbits, int inverse)

{

    FFTContext *s = av_malloc(sizeof(*s));



    if (s && ff_fft_init(s, nbits, inverse))

        av_freep(&s);



    return s;

}
