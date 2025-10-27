static void spr_write_40x_dbcr0 (void *opaque, int sprn)

{

    DisasContext *ctx = opaque;



    gen_op_store_40x_dbcr0();

    /* We must stop translation as we may have rebooted */

    RET_STOP(ctx);

}
