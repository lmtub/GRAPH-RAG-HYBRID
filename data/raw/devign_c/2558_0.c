static const TCGTargetOpDef *tcg_target_op_def(TCGOpcode op)

{

    static const TCGTargetOpDef r = { .args_ct_str = { "r" } };

    static const TCGTargetOpDef r_r = { .args_ct_str = { "r", "r" } };

    static const TCGTargetOpDef r_L = { .args_ct_str = { "r", "L" } };

    static const TCGTargetOpDef L_L = { .args_ct_str = { "L", "L" } };

    static const TCGTargetOpDef r_ri = { .args_ct_str = { "r", "ri" } };

    static const TCGTargetOpDef r_rC = { .args_ct_str = { "r", "rC" } };

    static const TCGTargetOpDef r_rZ = { .args_ct_str = { "r", "rZ" } };

    static const TCGTargetOpDef r_r_ri = { .args_ct_str = { "r", "r", "ri" } };

    static const TCGTargetOpDef r_0_ri = { .args_ct_str = { "r", "0", "ri" } };

    static const TCGTargetOpDef r_0_rI = { .args_ct_str = { "r", "0", "rI" } };

    static const TCGTargetOpDef r_0_rJ = { .args_ct_str = { "r", "0", "rJ" } };

    static const TCGTargetOpDef r_0_rO = { .args_ct_str = { "r", "0", "rO" } };

    static const TCGTargetOpDef r_0_rX = { .args_ct_str = { "r", "0", "rX" } };



    switch (op) {

    case INDEX_op_goto_ptr:

        return &r;



    case INDEX_op_ld8u_i32:

    case INDEX_op_ld8u_i64:

    case INDEX_op_ld8s_i32:

    case INDEX_op_ld8s_i64:

    case INDEX_op_ld16u_i32:

    case INDEX_op_ld16u_i64:

    case INDEX_op_ld16s_i32:

    case INDEX_op_ld16s_i64:

    case INDEX_op_ld_i32:

    case INDEX_op_ld32u_i64:

    case INDEX_op_ld32s_i64:

    case INDEX_op_ld_i64:

    case INDEX_op_st8_i32:

    case INDEX_op_st8_i64:

    case INDEX_op_st16_i32:

    case INDEX_op_st16_i64:

    case INDEX_op_st_i32:

    case INDEX_op_st32_i64:

    case INDEX_op_st_i64:

        return &r_r;



    case INDEX_op_add_i32:

    case INDEX_op_add_i64:

        return &r_r_ri;

    case INDEX_op_sub_i32:

    case INDEX_op_sub_i64:

        return &r_0_ri;



    case INDEX_op_mul_i32:

        /* If we have the general-instruction-extensions, then we have

           MULTIPLY SINGLE IMMEDIATE with a signed 32-bit, otherwise we

           have only MULTIPLY HALFWORD IMMEDIATE, with a signed 16-bit.  */

        return (s390_facilities & FACILITY_GEN_INST_EXT ? &r_0_ri : &r_0_rI);

    case INDEX_op_mul_i64:

        return (s390_facilities & FACILITY_GEN_INST_EXT ? &r_0_rJ : &r_0_rI);



    case INDEX_op_or_i32:

    case INDEX_op_or_i64:

        return &r_0_rO;

    case INDEX_op_xor_i32:

    case INDEX_op_xor_i64:

        return &r_0_rX;

    case INDEX_op_and_i32:

    case INDEX_op_and_i64:

        return &r_0_ri;



    case INDEX_op_shl_i32:

    case INDEX_op_shr_i32:

    case INDEX_op_sar_i32:

        return &r_0_ri;



    case INDEX_op_shl_i64:

    case INDEX_op_shr_i64:

    case INDEX_op_sar_i64:

        return &r_r_ri;



    case INDEX_op_rotl_i32:

    case INDEX_op_rotl_i64:

    case INDEX_op_rotr_i32:

    case INDEX_op_rotr_i64:

        return &r_r_ri;



    case INDEX_op_brcond_i32:

        /* Without EXT_IMM, only the LOAD AND TEST insn is available.  */

        return (s390_facilities & FACILITY_EXT_IMM ? &r_ri : &r_rZ);

    case INDEX_op_brcond_i64:

        return (s390_facilities & FACILITY_EXT_IMM ? &r_rC : &r_rZ);



    case INDEX_op_bswap16_i32:

    case INDEX_op_bswap16_i64:

    case INDEX_op_bswap32_i32:

    case INDEX_op_bswap32_i64:

    case INDEX_op_bswap64_i64:

    case INDEX_op_neg_i32:

    case INDEX_op_neg_i64:

    case INDEX_op_ext8s_i32:

    case INDEX_op_ext8s_i64:

    case INDEX_op_ext8u_i32:

    case INDEX_op_ext8u_i64:

    case INDEX_op_ext16s_i32:

    case INDEX_op_ext16s_i64:

    case INDEX_op_ext16u_i32:

    case INDEX_op_ext16u_i64:

    case INDEX_op_ext32s_i64:

    case INDEX_op_ext32u_i64:

    case INDEX_op_ext_i32_i64:

    case INDEX_op_extu_i32_i64:

    case INDEX_op_extract_i32:

    case INDEX_op_extract_i64:

        return &r_r;



    case INDEX_op_clz_i64:

        return &r_r_ri;



    case INDEX_op_qemu_ld_i32:

    case INDEX_op_qemu_ld_i64:

        return &r_L;

    case INDEX_op_qemu_st_i64:

    case INDEX_op_qemu_st_i32:

        return &L_L;



    case INDEX_op_deposit_i32:

    case INDEX_op_deposit_i64:

        {

            static const TCGTargetOpDef dep

                = { .args_ct_str = { "r", "rZ", "r" } };

            return &dep;

        }

    case INDEX_op_setcond_i32:

    case INDEX_op_setcond_i64:

        {

            /* Without EXT_IMM, only the LOAD AND TEST insn is available.  */

            static const TCGTargetOpDef setc_z

                = { .args_ct_str = { "r", "r", "rZ" } };

            static const TCGTargetOpDef setc_c

                = { .args_ct_str = { "r", "r", "rC" } };

            return (s390_facilities & FACILITY_EXT_IMM ? &setc_c : &setc_z);

        }

    case INDEX_op_movcond_i32:

    case INDEX_op_movcond_i64:

        {

            /* Without EXT_IMM, only the LOAD AND TEST insn is available.  */

            static const TCGTargetOpDef movc_z

                = { .args_ct_str = { "r", "r", "rZ", "r", "0" } };

            static const TCGTargetOpDef movc_c

                = { .args_ct_str = { "r", "r", "rC", "r", "0" } };

            return (s390_facilities & FACILITY_EXT_IMM ? &movc_c : &movc_z);

        }

    case INDEX_op_div2_i32:

    case INDEX_op_div2_i64:

    case INDEX_op_divu2_i32:

    case INDEX_op_divu2_i64:

        {

            static const TCGTargetOpDef div2

                = { .args_ct_str = { "b", "a", "0", "1", "r" } };

            return &div2;

        }

    case INDEX_op_mulu2_i64:

        {

            static const TCGTargetOpDef mul2

                = { .args_ct_str = { "b", "a", "0", "r" } };

            return &mul2;

        }

    case INDEX_op_add2_i32:

    case INDEX_op_add2_i64:

    case INDEX_op_sub2_i32:

    case INDEX_op_sub2_i64:

        {

            static const TCGTargetOpDef arith2

                = { .args_ct_str = { "r", "r", "0", "1", "rA", "r" } };

            return &arith2;

        }



    default:

        break;

    }

    return NULL;

}
