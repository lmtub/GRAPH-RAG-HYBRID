static void spr_write_dbatu (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_dbatu((sprn - SPR_DBAT0U) / 2);

    RET_STOP(ctx);

}
