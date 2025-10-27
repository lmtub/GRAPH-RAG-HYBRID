static void tcg_out_op(TCGContext *s, TCGOpcode opc,

                       const TCGArg args[TCG_MAX_OP_ARGS],

                       const int const_args[TCG_MAX_OP_ARGS])

{

    /* 99% of the time, we can signal the use of extension registers

       by looking to see if the opcode handles 64-bit data.  */

    TCGType ext = (tcg_op_defs[opc].flags & TCG_OPF_64BIT) != 0;



    /* Hoist the loads of the most common arguments.  */

    TCGArg a0 = args[0];

    TCGArg a1 = args[1];

    TCGArg a2 = args[2];

    int c2 = const_args[2];



    /* Some operands are defined with "rZ" constraint, a register or

       the zero register.  These need not actually test args[I] == 0.  */

#define REG0(I)  (const_args[I] ? TCG_REG_XZR : (TCGReg)args[I])



    switch (opc) {

    case INDEX_op_exit_tb:

        tcg_out_movi(s, TCG_TYPE_I64, TCG_REG_X0, a0);

        tcg_out_goto(s, tb_ret_addr);

        break;



    case INDEX_op_goto_tb:

#ifndef USE_DIRECT_JUMP

#error "USE_DIRECT_JUMP required for aarch64"

#endif

        assert(s->tb_jmp_offset != NULL); /* consistency for USE_DIRECT_JUMP */

        s->tb_jmp_offset[a0] = tcg_current_code_size(s);

        /* actual branch destination will be patched by

           aarch64_tb_set_jmp_target later, beware retranslation. */

        tcg_out_goto_noaddr(s);

        s->tb_next_offset[a0] = tcg_current_code_size(s);

        break;



    case INDEX_op_br:

        tcg_out_goto_label(s, arg_label(a0));

        break;



    case INDEX_op_ld8u_i32:

    case INDEX_op_ld8u_i64:

        tcg_out_ldst(s, I3312_LDRB, a0, a1, a2);

        break;

    case INDEX_op_ld8s_i32:

        tcg_out_ldst(s, I3312_LDRSBW, a0, a1, a2);

        break;

    case INDEX_op_ld8s_i64:

        tcg_out_ldst(s, I3312_LDRSBX, a0, a1, a2);

        break;

    case INDEX_op_ld16u_i32:

    case INDEX_op_ld16u_i64:

        tcg_out_ldst(s, I3312_LDRH, a0, a1, a2);

        break;

    case INDEX_op_ld16s_i32:

        tcg_out_ldst(s, I3312_LDRSHW, a0, a1, a2);

        break;

    case INDEX_op_ld16s_i64:

        tcg_out_ldst(s, I3312_LDRSHX, a0, a1, a2);

        break;

    case INDEX_op_ld_i32:

    case INDEX_op_ld32u_i64:

        tcg_out_ldst(s, I3312_LDRW, a0, a1, a2);

        break;

    case INDEX_op_ld32s_i64:

        tcg_out_ldst(s, I3312_LDRSWX, a0, a1, a2);

        break;

    case INDEX_op_ld_i64:

        tcg_out_ldst(s, I3312_LDRX, a0, a1, a2);

        break;



    case INDEX_op_st8_i32:

    case INDEX_op_st8_i64:

        tcg_out_ldst(s, I3312_STRB, REG0(0), a1, a2);

        break;

    case INDEX_op_st16_i32:

    case INDEX_op_st16_i64:

        tcg_out_ldst(s, I3312_STRH, REG0(0), a1, a2);

        break;

    case INDEX_op_st_i32:

    case INDEX_op_st32_i64:

        tcg_out_ldst(s, I3312_STRW, REG0(0), a1, a2);

        break;

    case INDEX_op_st_i64:

        tcg_out_ldst(s, I3312_STRX, REG0(0), a1, a2);

        break;



    case INDEX_op_add_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_add_i64:

        if (c2) {

            tcg_out_addsubi(s, ext, a0, a1, a2);

        } else {

            tcg_out_insn(s, 3502, ADD, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_sub_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_sub_i64:

        if (c2) {

            tcg_out_addsubi(s, ext, a0, a1, -a2);

        } else {

            tcg_out_insn(s, 3502, SUB, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_neg_i64:

    case INDEX_op_neg_i32:

        tcg_out_insn(s, 3502, SUB, ext, a0, TCG_REG_XZR, a1);

        break;



    case INDEX_op_and_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_and_i64:

        if (c2) {

            tcg_out_logicali(s, I3404_ANDI, ext, a0, a1, a2);

        } else {

            tcg_out_insn(s, 3510, AND, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_andc_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_andc_i64:

        if (c2) {

            tcg_out_logicali(s, I3404_ANDI, ext, a0, a1, ~a2);

        } else {

            tcg_out_insn(s, 3510, BIC, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_or_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_or_i64:

        if (c2) {

            tcg_out_logicali(s, I3404_ORRI, ext, a0, a1, a2);

        } else {

            tcg_out_insn(s, 3510, ORR, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_orc_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_orc_i64:

        if (c2) {

            tcg_out_logicali(s, I3404_ORRI, ext, a0, a1, ~a2);

        } else {

            tcg_out_insn(s, 3510, ORN, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_xor_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_xor_i64:

        if (c2) {

            tcg_out_logicali(s, I3404_EORI, ext, a0, a1, a2);

        } else {

            tcg_out_insn(s, 3510, EOR, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_eqv_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_eqv_i64:

        if (c2) {

            tcg_out_logicali(s, I3404_EORI, ext, a0, a1, ~a2);

        } else {

            tcg_out_insn(s, 3510, EON, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_not_i64:

    case INDEX_op_not_i32:

        tcg_out_insn(s, 3510, ORN, ext, a0, TCG_REG_XZR, a1);

        break;



    case INDEX_op_mul_i64:

    case INDEX_op_mul_i32:

        tcg_out_insn(s, 3509, MADD, ext, a0, a1, a2, TCG_REG_XZR);

        break;



    case INDEX_op_div_i64:

    case INDEX_op_div_i32:

        tcg_out_insn(s, 3508, SDIV, ext, a0, a1, a2);

        break;

    case INDEX_op_divu_i64:

    case INDEX_op_divu_i32:

        tcg_out_insn(s, 3508, UDIV, ext, a0, a1, a2);

        break;



    case INDEX_op_rem_i64:

    case INDEX_op_rem_i32:

        tcg_out_insn(s, 3508, SDIV, ext, TCG_REG_TMP, a1, a2);

        tcg_out_insn(s, 3509, MSUB, ext, a0, TCG_REG_TMP, a2, a1);

        break;

    case INDEX_op_remu_i64:

    case INDEX_op_remu_i32:

        tcg_out_insn(s, 3508, UDIV, ext, TCG_REG_TMP, a1, a2);

        tcg_out_insn(s, 3509, MSUB, ext, a0, TCG_REG_TMP, a2, a1);

        break;



    case INDEX_op_shl_i64:

    case INDEX_op_shl_i32:

        if (c2) {

            tcg_out_shl(s, ext, a0, a1, a2);

        } else {

            tcg_out_insn(s, 3508, LSLV, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_shr_i64:

    case INDEX_op_shr_i32:

        if (c2) {

            tcg_out_shr(s, ext, a0, a1, a2);

        } else {

            tcg_out_insn(s, 3508, LSRV, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_sar_i64:

    case INDEX_op_sar_i32:

        if (c2) {

            tcg_out_sar(s, ext, a0, a1, a2);

        } else {

            tcg_out_insn(s, 3508, ASRV, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_rotr_i64:

    case INDEX_op_rotr_i32:

        if (c2) {

            tcg_out_rotr(s, ext, a0, a1, a2);

        } else {

            tcg_out_insn(s, 3508, RORV, ext, a0, a1, a2);

        }

        break;



    case INDEX_op_rotl_i64:

    case INDEX_op_rotl_i32:

        if (c2) {

            tcg_out_rotl(s, ext, a0, a1, a2);

        } else {

            tcg_out_insn(s, 3502, SUB, 0, TCG_REG_TMP, TCG_REG_XZR, a2);

            tcg_out_insn(s, 3508, RORV, ext, a0, a1, TCG_REG_TMP);

        }

        break;



    case INDEX_op_brcond_i32:

        a1 = (int32_t)a1;

        /* FALLTHRU */

    case INDEX_op_brcond_i64:

        tcg_out_brcond(s, ext, a2, a0, a1, const_args[1], arg_label(args[3]));

        break;



    case INDEX_op_setcond_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_setcond_i64:

        tcg_out_cmp(s, ext, a1, a2, c2);

        /* Use CSET alias of CSINC Wd, WZR, WZR, invert(cond).  */

        tcg_out_insn(s, 3506, CSINC, TCG_TYPE_I32, a0, TCG_REG_XZR,

                     TCG_REG_XZR, tcg_invert_cond(args[3]));

        break;



    case INDEX_op_movcond_i32:

        a2 = (int32_t)a2;

        /* FALLTHRU */

    case INDEX_op_movcond_i64:

        tcg_out_cmp(s, ext, a1, a2, c2);

        tcg_out_insn(s, 3506, CSEL, ext, a0, REG0(3), REG0(4), args[5]);

        break;



    case INDEX_op_qemu_ld_i32:

    case INDEX_op_qemu_ld_i64:

        tcg_out_qemu_ld(s, a0, a1, a2, ext);

        break;

    case INDEX_op_qemu_st_i32:

    case INDEX_op_qemu_st_i64:

        tcg_out_qemu_st(s, REG0(0), a1, a2);

        break;



    case INDEX_op_bswap64_i64:

        tcg_out_rev64(s, a0, a1);

        break;

    case INDEX_op_bswap32_i64:

    case INDEX_op_bswap32_i32:

        tcg_out_rev32(s, a0, a1);

        break;

    case INDEX_op_bswap16_i64:

    case INDEX_op_bswap16_i32:

        tcg_out_rev16(s, a0, a1);

        break;



    case INDEX_op_ext8s_i64:

    case INDEX_op_ext8s_i32:

        tcg_out_sxt(s, ext, MO_8, a0, a1);

        break;

    case INDEX_op_ext16s_i64:

    case INDEX_op_ext16s_i32:

        tcg_out_sxt(s, ext, MO_16, a0, a1);

        break;

    case INDEX_op_ext_i32_i64:

    case INDEX_op_ext32s_i64:

        tcg_out_sxt(s, TCG_TYPE_I64, MO_32, a0, a1);

        break;

    case INDEX_op_ext8u_i64:

    case INDEX_op_ext8u_i32:

        tcg_out_uxt(s, MO_8, a0, a1);

        break;

    case INDEX_op_ext16u_i64:

    case INDEX_op_ext16u_i32:

        tcg_out_uxt(s, MO_16, a0, a1);

        break;

    case INDEX_op_extu_i32_i64:

    case INDEX_op_ext32u_i64:

        tcg_out_movr(s, TCG_TYPE_I32, a0, a1);

        break;



    case INDEX_op_deposit_i64:

    case INDEX_op_deposit_i32:

        tcg_out_dep(s, ext, a0, REG0(2), args[3], args[4]);

        break;



    case INDEX_op_add2_i32:

        tcg_out_addsub2(s, TCG_TYPE_I32, a0, a1, REG0(2), REG0(3),

                        (int32_t)args[4], args[5], const_args[4],

                        const_args[5], false);

        break;

    case INDEX_op_add2_i64:

        tcg_out_addsub2(s, TCG_TYPE_I64, a0, a1, REG0(2), REG0(3), args[4],

                        args[5], const_args[4], const_args[5], false);

        break;

    case INDEX_op_sub2_i32:

        tcg_out_addsub2(s, TCG_TYPE_I32, a0, a1, REG0(2), REG0(3),

                        (int32_t)args[4], args[5], const_args[4],

                        const_args[5], true);

        break;

    case INDEX_op_sub2_i64:

        tcg_out_addsub2(s, TCG_TYPE_I64, a0, a1, REG0(2), REG0(3), args[4],

                        args[5], const_args[4], const_args[5], true);

        break;



    case INDEX_op_muluh_i64:

        tcg_out_insn(s, 3508, UMULH, TCG_TYPE_I64, a0, a1, a2);

        break;

    case INDEX_op_mulsh_i64:

        tcg_out_insn(s, 3508, SMULH, TCG_TYPE_I64, a0, a1, a2);

        break;



    case INDEX_op_mov_i32:  /* Always emitted via tcg_out_mov.  */

    case INDEX_op_mov_i64:

    case INDEX_op_movi_i32: /* Always emitted via tcg_out_movi.  */

    case INDEX_op_movi_i64:

    case INDEX_op_call:     /* Always emitted via tcg_out_call.  */

    default:

        tcg_abort();

    }



#undef REG0

}
