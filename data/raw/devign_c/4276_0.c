static void spr_write_601_ubatl (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_601_batl((sprn - SPR_IBAT0L) / 2);

    RET_STOP(ctx);

}
