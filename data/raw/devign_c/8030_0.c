av_cold void ff_synth_filter_init(SynthFilterContext *c)

{

    c->synth_filter_float = synth_filter_float;



    if (ARCH_ARM) ff_synth_filter_init_arm(c);

    if (ARCH_X86) ff_synth_filter_init_x86(c);

}
