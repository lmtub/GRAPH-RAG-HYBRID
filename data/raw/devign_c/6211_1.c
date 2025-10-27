av_cold void ff_mlpdsp_init(MLPDSPContext *c)

{

    c->mlp_filter_channel = mlp_filter_channel;



    if (ARCH_X86)

        ff_mlpdsp_init_x86(c);

}