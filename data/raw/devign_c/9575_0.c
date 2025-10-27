static void gen_store_exclusive(DisasContext *s, int rd, int rt, int rt2,

                                TCGv_i32 addr, int size)

{

    TCGv_i32 tmp;

    TCGv_i64 val64, extaddr;

    int done_label;

    int fail_label;



    /* if (env->exclusive_addr == addr && env->exclusive_val == [addr]) {

         [addr] = {Rt};

         {Rd} = 0;

       } else {

         {Rd} = 1;

       } */

    fail_label = gen_new_label();

    done_label = gen_new_label();

    extaddr = tcg_temp_new_i64();

    tcg_gen_extu_i32_i64(extaddr, addr);

    tcg_gen_brcond_i64(TCG_COND_NE, extaddr, cpu_exclusive_addr, fail_label);

    tcg_temp_free_i64(extaddr);



    tmp = tcg_temp_new_i32();

    switch (size) {

    case 0:

        gen_aa32_ld8u(tmp, addr, get_mem_index(s));

        break;

    case 1:

        gen_aa32_ld16u(tmp, addr, get_mem_index(s));

        break;

    case 2:

    case 3:

        gen_aa32_ld32u(tmp, addr, get_mem_index(s));

        break;

    default:

        abort();

    }



    val64 = tcg_temp_new_i64();

    if (size == 3) {

        TCGv_i32 tmp2 = tcg_temp_new_i32();

        TCGv_i32 tmp3 = tcg_temp_new_i32();

        tcg_gen_addi_i32(tmp2, addr, 4);

        gen_aa32_ld32u(tmp3, tmp2, get_mem_index(s));

        tcg_temp_free_i32(tmp2);

        tcg_gen_concat_i32_i64(val64, tmp, tmp3);

        tcg_temp_free_i32(tmp3);

    } else {

        tcg_gen_extu_i32_i64(val64, tmp);

    }

    tcg_temp_free_i32(tmp);



    tcg_gen_brcond_i64(TCG_COND_NE, val64, cpu_exclusive_val, fail_label);

    tcg_temp_free_i64(val64);



    tmp = load_reg(s, rt);

    switch (size) {

    case 0:

        gen_aa32_st8(tmp, addr, get_mem_index(s));

        break;

    case 1:

        gen_aa32_st16(tmp, addr, get_mem_index(s));

        break;

    case 2:

    case 3:

        gen_aa32_st32(tmp, addr, get_mem_index(s));

        break;

    default:

        abort();

    }

    tcg_temp_free_i32(tmp);

    if (size == 3) {

        tcg_gen_addi_i32(addr, addr, 4);

        tmp = load_reg(s, rt2);

        gen_aa32_st32(tmp, addr, get_mem_index(s));

        tcg_temp_free_i32(tmp);

    }

    tcg_gen_movi_i32(cpu_R[rd], 0);

    tcg_gen_br(done_label);

    gen_set_label(fail_label);

    tcg_gen_movi_i32(cpu_R[rd], 1);

    gen_set_label(done_label);

    tcg_gen_movi_i64(cpu_exclusive_addr, -1);

}
