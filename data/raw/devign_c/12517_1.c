static void gen_waiti(DisasContext *dc, uint32_t imm4)

{

    TCGv_i32 pc = tcg_const_i32(dc->next_pc);

    TCGv_i32 intlevel = tcg_const_i32(imm4);



    if (dc->tb->cflags & CF_USE_ICOUNT) {

        gen_io_start();

    }

    gen_helper_waiti(cpu_env, pc, intlevel);

    if (dc->tb->cflags & CF_USE_ICOUNT) {

        gen_io_end();

    }

    tcg_temp_free(pc);

    tcg_temp_free(intlevel);

    gen_jumpi_check_loop_end(dc, 0);

}
