av_cold void ff_lpc_end(LPCContext *s)

{

    av_freep(&s->windowed_samples);

}
