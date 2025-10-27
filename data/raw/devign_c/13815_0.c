static void gen_srs(DisasContext *s,

                    uint32_t mode, uint32_t amode, bool writeback)

{

    int32_t offset;

    TCGv_i32 addr = tcg_temp_new_i32();

    TCGv_i32 tmp = tcg_const_i32(mode);

    gen_helper_get_r13_banked(addr, cpu_env, tmp);

    tcg_temp_free_i32(tmp);

    switch (amode) {

    case 0: /* DA */

        offset = -4;

        break;

    case 1: /* IA */

        offset = 0;

        break;

    case 2: /* DB */

        offset = -8;

        break;

    case 3: /* IB */

        offset = 4;

        break;

    default:

        abort();

    }

    tcg_gen_addi_i32(addr, addr, offset);

    tmp = load_reg(s, 14);

    gen_aa32_st32(tmp, addr, get_mem_index(s));

    tcg_temp_free_i32(tmp);

    tmp = load_cpu_field(spsr);

    tcg_gen_addi_i32(addr, addr, 4);

    gen_aa32_st32(tmp, addr, get_mem_index(s));

    tcg_temp_free_i32(tmp);

    if (writeback) {

        switch (amode) {

        case 0:

            offset = -8;

            break;

        case 1:

            offset = 4;

            break;

        case 2:

            offset = -4;

            break;

        case 3:

            offset = 0;

            break;

        default:

            abort();

        }

        tcg_gen_addi_i32(addr, addr, offset);

        tmp = tcg_const_i32(mode);

        gen_helper_set_r13_banked(cpu_env, tmp, addr);

        tcg_temp_free_i32(tmp);

    }

    tcg_temp_free_i32(addr);

}
