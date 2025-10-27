void ff_mlp_init_x86(DSPContext* c, AVCodecContext *avctx)

{

#if HAVE_7REGS && HAVE_TEN_OPERANDS

    c->mlp_filter_channel = mlp_filter_channel_x86;

#endif

}
