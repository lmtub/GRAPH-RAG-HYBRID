static void do_fp_ld(DisasContext *s, int destidx, TCGv_i64 tcg_addr, int size)

{

    /* This always zero-extends and writes to a full 128 bit wide vector */

    TCGv_i64 tmplo = tcg_temp_new_i64();

    TCGv_i64 tmphi;



    if (size < 4) {

        TCGMemOp memop = MO_TE + size;

        tmphi = tcg_const_i64(0);

        tcg_gen_qemu_ld_i64(tmplo, tcg_addr, get_mem_index(s), memop);

    } else {

        TCGv_i64 tcg_hiaddr;

        tmphi = tcg_temp_new_i64();

        tcg_hiaddr = tcg_temp_new_i64();



        tcg_gen_qemu_ld_i64(tmplo, tcg_addr, get_mem_index(s), MO_TEQ);

        tcg_gen_addi_i64(tcg_hiaddr, tcg_addr, 8);

        tcg_gen_qemu_ld_i64(tmphi, tcg_hiaddr, get_mem_index(s), MO_TEQ);

        tcg_temp_free_i64(tcg_hiaddr);

    }



    tcg_gen_st_i64(tmplo, cpu_env, fp_reg_offset(destidx, MO_64));

    tcg_gen_st_i64(tmphi, cpu_env, fp_reg_hi_offset(destidx));



    tcg_temp_free_i64(tmplo);

    tcg_temp_free_i64(tmphi);

}
