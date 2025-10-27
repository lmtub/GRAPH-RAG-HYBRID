static void spr_write_tbl (DisasContext *ctx, int sprn, int gprn)

{

    if (use_icount) {

        gen_io_start();

    }

    gen_helper_store_tbl(cpu_env, cpu_gpr[gprn]);

    if (use_icount) {

        gen_io_end();

        gen_stop_exception(ctx);

    }

}
