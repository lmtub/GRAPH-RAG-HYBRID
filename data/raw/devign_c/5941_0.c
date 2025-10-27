av_cold void ff_vp8dsp_init_arm(VP8DSPContext *dsp)

{

    int cpu_flags = av_get_cpu_flags();



    if (have_armv6(cpu_flags))

        ff_vp8dsp_init_armv6(dsp);

    if (have_neon(cpu_flags))

        ff_vp8dsp_init_neon(dsp);

}
