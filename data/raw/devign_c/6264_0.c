static void spr_write_ibatu (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_ibatu((sprn - SPR_IBAT0U) / 2);

    RET_STOP(ctx);

}
