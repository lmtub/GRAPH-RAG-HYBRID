static void read_vec_element_i32(DisasContext *s, TCGv_i32 tcg_dest, int srcidx,

                                 int element, TCGMemOp memop)

{

    int vect_off = vec_reg_offset(srcidx, element, memop & MO_SIZE);

    switch (memop) {

    case MO_8:

        tcg_gen_ld8u_i32(tcg_dest, cpu_env, vect_off);

        break;

    case MO_16:

        tcg_gen_ld16u_i32(tcg_dest, cpu_env, vect_off);

        break;

    case MO_8|MO_SIGN:

        tcg_gen_ld8s_i32(tcg_dest, cpu_env, vect_off);

        break;

    case MO_16|MO_SIGN:

        tcg_gen_ld16s_i32(tcg_dest, cpu_env, vect_off);

        break;

    case MO_32:

    case MO_32|MO_SIGN:

        tcg_gen_ld_i32(tcg_dest, cpu_env, vect_off);

        break;

    default:

        g_assert_not_reached();

    }

}
