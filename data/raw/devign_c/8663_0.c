static void spr_write_dbatl (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_dbatl((sprn - SPR_DBAT0L) / 2);

    RET_STOP(ctx);

}
