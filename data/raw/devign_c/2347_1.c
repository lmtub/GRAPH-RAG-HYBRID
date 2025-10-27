av_cold void ff_mlpdsp_init_arm(MLPDSPContext *c)

{

    int cpu_flags = av_get_cpu_flags();



    if (have_armv5te(cpu_flags)) {

        c->mlp_filter_channel = ff_mlp_filter_channel_arm;


    }

}