static inline void tcg_out_op(TCGContext *s, TCGOpcode opc,

                const TCGArg *args, const int *const_args)

{

    int c;



    switch (opc) {

    case INDEX_op_exit_tb:

        {

            uint8_t *ld_ptr = s->code_ptr;

            if (args[0] >> 8)

                tcg_out_ld32_12(s, COND_AL, TCG_REG_R0, TCG_REG_PC, 0);

            else

                tcg_out_dat_imm(s, COND_AL, ARITH_MOV, TCG_REG_R0, 0, args[0]);

            tcg_out_goto(s, COND_AL, (tcg_target_ulong) tb_ret_addr);

            if (args[0] >> 8) {

                *ld_ptr = (uint8_t) (s->code_ptr - ld_ptr) - 8;

                tcg_out32(s, args[0]);

            }

        }

        break;

    case INDEX_op_goto_tb:

        if (s->tb_jmp_offset) {

            /* Direct jump method */

#if defined(USE_DIRECT_JUMP)

            s->tb_jmp_offset[args[0]] = s->code_ptr - s->code_buf;

            tcg_out_b_noaddr(s, COND_AL);

#else

            tcg_out_ld32_12(s, COND_AL, TCG_REG_PC, TCG_REG_PC, -4);

            s->tb_jmp_offset[args[0]] = s->code_ptr - s->code_buf;

            tcg_out32(s, 0);

#endif

        } else {

            /* Indirect jump method */

#if 1

            c = (int) (s->tb_next + args[0]) - ((int) s->code_ptr + 8);

            if (c > 0xfff || c < -0xfff) {

                tcg_out_movi32(s, COND_AL, TCG_REG_R0,

                                (tcg_target_long) (s->tb_next + args[0]));

                tcg_out_ld32_12(s, COND_AL, TCG_REG_PC, TCG_REG_R0, 0);

            } else

                tcg_out_ld32_12(s, COND_AL, TCG_REG_PC, TCG_REG_PC, c);

#else

            tcg_out_ld32_12(s, COND_AL, TCG_REG_R0, TCG_REG_PC, 0);

            tcg_out_ld32_12(s, COND_AL, TCG_REG_PC, TCG_REG_R0, 0);

            tcg_out32(s, (tcg_target_long) (s->tb_next + args[0]));

#endif

        }

        s->tb_next_offset[args[0]] = s->code_ptr - s->code_buf;

        break;

    case INDEX_op_call:

        if (const_args[0])

            tcg_out_call(s, args[0]);

        else

            tcg_out_callr(s, COND_AL, args[0]);

        break;

    case INDEX_op_br:

        tcg_out_goto_label(s, COND_AL, args[0]);

        break;



    case INDEX_op_ld8u_i32:

        tcg_out_ld8u(s, COND_AL, args[0], args[1], args[2]);

        break;

    case INDEX_op_ld8s_i32:

        tcg_out_ld8s(s, COND_AL, args[0], args[1], args[2]);

        break;

    case INDEX_op_ld16u_i32:

        tcg_out_ld16u(s, COND_AL, args[0], args[1], args[2]);

        break;

    case INDEX_op_ld16s_i32:

        tcg_out_ld16s(s, COND_AL, args[0], args[1], args[2]);

        break;

    case INDEX_op_ld_i32:

        tcg_out_ld32u(s, COND_AL, args[0], args[1], args[2]);

        break;

    case INDEX_op_st8_i32:

        tcg_out_st8(s, COND_AL, args[0], args[1], args[2]);

        break;

    case INDEX_op_st16_i32:

        tcg_out_st16(s, COND_AL, args[0], args[1], args[2]);

        break;

    case INDEX_op_st_i32:

        tcg_out_st32(s, COND_AL, args[0], args[1], args[2]);

        break;



    case INDEX_op_mov_i32:

        tcg_out_dat_reg(s, COND_AL, ARITH_MOV,

                        args[0], 0, args[1], SHIFT_IMM_LSL(0));

        break;

    case INDEX_op_movi_i32:

        tcg_out_movi32(s, COND_AL, args[0], args[1]);

        break;

    case INDEX_op_movcond_i32:

        /* Constraints mean that v2 is always in the same register as dest,

         * so we only need to do "if condition passed, move v1 to dest".

         */

        tcg_out_dat_rI(s, COND_AL, ARITH_CMP, 0,

                       args[1], args[2], const_args[2]);

        tcg_out_dat_rI(s, tcg_cond_to_arm_cond[args[5]],

                       ARITH_MOV, args[0], 0, args[3], const_args[3]);

        break;

    case INDEX_op_add_i32:

        c = ARITH_ADD;

        goto gen_arith;

    case INDEX_op_sub_i32:

        c = ARITH_SUB;

        goto gen_arith;

    case INDEX_op_and_i32:

        tcg_out_dat_rIK(s, COND_AL, ARITH_AND, ARITH_BIC,

                        args[0], args[1], args[2], const_args[2]);

        break;

    case INDEX_op_andc_i32:

        tcg_out_dat_rIK(s, COND_AL, ARITH_BIC, ARITH_AND,

                        args[0], args[1], args[2], const_args[2]);

        break;

    case INDEX_op_or_i32:

        c = ARITH_ORR;

        goto gen_arith;

    case INDEX_op_xor_i32:

        c = ARITH_EOR;

        /* Fall through.  */

    gen_arith:

        tcg_out_dat_rI(s, COND_AL, c, args[0], args[1], args[2], const_args[2]);

        break;

    case INDEX_op_add2_i32:

        tcg_out_dat_reg2(s, COND_AL, ARITH_ADD, ARITH_ADC,

                        args[0], args[1], args[2], args[3],

                        args[4], args[5], SHIFT_IMM_LSL(0));

        break;

    case INDEX_op_sub2_i32:

        tcg_out_dat_reg2(s, COND_AL, ARITH_SUB, ARITH_SBC,

                        args[0], args[1], args[2], args[3],

                        args[4], args[5], SHIFT_IMM_LSL(0));

        break;

    case INDEX_op_neg_i32:

        tcg_out_dat_imm(s, COND_AL, ARITH_RSB, args[0], args[1], 0);

        break;

    case INDEX_op_not_i32:

        tcg_out_dat_reg(s, COND_AL,

                        ARITH_MVN, args[0], 0, args[1], SHIFT_IMM_LSL(0));

        break;

    case INDEX_op_mul_i32:

        tcg_out_mul32(s, COND_AL, args[0], args[1], args[2]);

        break;

    case INDEX_op_mulu2_i32:

        tcg_out_umull32(s, COND_AL, args[0], args[1], args[2], args[3]);

        break;

    case INDEX_op_muls2_i32:

        tcg_out_smull32(s, COND_AL, args[0], args[1], args[2], args[3]);

        break;

    /* XXX: Perhaps args[2] & 0x1f is wrong */

    case INDEX_op_shl_i32:

        c = const_args[2] ?

                SHIFT_IMM_LSL(args[2] & 0x1f) : SHIFT_REG_LSL(args[2]);

        goto gen_shift32;

    case INDEX_op_shr_i32:

        c = const_args[2] ? (args[2] & 0x1f) ? SHIFT_IMM_LSR(args[2] & 0x1f) :

                SHIFT_IMM_LSL(0) : SHIFT_REG_LSR(args[2]);

        goto gen_shift32;

    case INDEX_op_sar_i32:

        c = const_args[2] ? (args[2] & 0x1f) ? SHIFT_IMM_ASR(args[2] & 0x1f) :

                SHIFT_IMM_LSL(0) : SHIFT_REG_ASR(args[2]);

        goto gen_shift32;

    case INDEX_op_rotr_i32:

        c = const_args[2] ? (args[2] & 0x1f) ? SHIFT_IMM_ROR(args[2] & 0x1f) :

                SHIFT_IMM_LSL(0) : SHIFT_REG_ROR(args[2]);

        /* Fall through.  */

    gen_shift32:

        tcg_out_dat_reg(s, COND_AL, ARITH_MOV, args[0], 0, args[1], c);

        break;



    case INDEX_op_rotl_i32:

        if (const_args[2]) {

            tcg_out_dat_reg(s, COND_AL, ARITH_MOV, args[0], 0, args[1],

                            ((0x20 - args[2]) & 0x1f) ?

                            SHIFT_IMM_ROR((0x20 - args[2]) & 0x1f) :

                            SHIFT_IMM_LSL(0));

        } else {

            tcg_out_dat_imm(s, COND_AL, ARITH_RSB, TCG_REG_R8, args[1], 0x20);

            tcg_out_dat_reg(s, COND_AL, ARITH_MOV, args[0], 0, args[1],

                            SHIFT_REG_ROR(TCG_REG_R8));

        }

        break;



    case INDEX_op_brcond_i32:

        tcg_out_dat_rI(s, COND_AL, ARITH_CMP, 0,

                       args[0], args[1], const_args[1]);

        tcg_out_goto_label(s, tcg_cond_to_arm_cond[args[2]], args[3]);

        break;

    case INDEX_op_brcond2_i32:

        /* The resulting conditions are:

         * TCG_COND_EQ    -->  a0 == a2 && a1 == a3,

         * TCG_COND_NE    --> (a0 != a2 && a1 == a3) ||  a1 != a3,

         * TCG_COND_LT(U) --> (a0 <  a2 && a1 == a3) ||  a1 <  a3,

         * TCG_COND_GE(U) --> (a0 >= a2 && a1 == a3) || (a1 >= a3 && a1 != a3),

         * TCG_COND_LE(U) --> (a0 <= a2 && a1 == a3) || (a1 <= a3 && a1 != a3),

         * TCG_COND_GT(U) --> (a0 >  a2 && a1 == a3) ||  a1 >  a3,

         */

        tcg_out_dat_reg(s, COND_AL, ARITH_CMP, 0,

                        args[1], args[3], SHIFT_IMM_LSL(0));

        tcg_out_dat_reg(s, COND_EQ, ARITH_CMP, 0,

                        args[0], args[2], SHIFT_IMM_LSL(0));

        tcg_out_goto_label(s, tcg_cond_to_arm_cond[args[4]], args[5]);

        break;

    case INDEX_op_setcond_i32:

        tcg_out_dat_rI(s, COND_AL, ARITH_CMP, 0,

                       args[1], args[2], const_args[2]);

        tcg_out_dat_imm(s, tcg_cond_to_arm_cond[args[3]],

                        ARITH_MOV, args[0], 0, 1);

        tcg_out_dat_imm(s, tcg_cond_to_arm_cond[tcg_invert_cond(args[3])],

                        ARITH_MOV, args[0], 0, 0);

        break;

    case INDEX_op_setcond2_i32:

        /* See brcond2_i32 comment */

        tcg_out_dat_reg(s, COND_AL, ARITH_CMP, 0,

                        args[2], args[4], SHIFT_IMM_LSL(0));

        tcg_out_dat_reg(s, COND_EQ, ARITH_CMP, 0,

                        args[1], args[3], SHIFT_IMM_LSL(0));

        tcg_out_dat_imm(s, tcg_cond_to_arm_cond[args[5]],

                        ARITH_MOV, args[0], 0, 1);

        tcg_out_dat_imm(s, tcg_cond_to_arm_cond[tcg_invert_cond(args[5])],

                        ARITH_MOV, args[0], 0, 0);

        break;



    case INDEX_op_qemu_ld8u:

        tcg_out_qemu_ld(s, args, 0);

        break;

    case INDEX_op_qemu_ld8s:

        tcg_out_qemu_ld(s, args, 0 | 4);

        break;

    case INDEX_op_qemu_ld16u:

        tcg_out_qemu_ld(s, args, 1);

        break;

    case INDEX_op_qemu_ld16s:

        tcg_out_qemu_ld(s, args, 1 | 4);

        break;

    case INDEX_op_qemu_ld32:

        tcg_out_qemu_ld(s, args, 2);

        break;

    case INDEX_op_qemu_ld64:

        tcg_out_qemu_ld(s, args, 3);

        break;



    case INDEX_op_qemu_st8:

        tcg_out_qemu_st(s, args, 0);

        break;

    case INDEX_op_qemu_st16:

        tcg_out_qemu_st(s, args, 1);

        break;

    case INDEX_op_qemu_st32:

        tcg_out_qemu_st(s, args, 2);

        break;

    case INDEX_op_qemu_st64:

        tcg_out_qemu_st(s, args, 3);

        break;



    case INDEX_op_bswap16_i32:

        tcg_out_bswap16(s, COND_AL, args[0], args[1]);

        break;

    case INDEX_op_bswap32_i32:

        tcg_out_bswap32(s, COND_AL, args[0], args[1]);

        break;



    case INDEX_op_ext8s_i32:

        tcg_out_ext8s(s, COND_AL, args[0], args[1]);

        break;

    case INDEX_op_ext16s_i32:

        tcg_out_ext16s(s, COND_AL, args[0], args[1]);

        break;

    case INDEX_op_ext16u_i32:

        tcg_out_ext16u(s, COND_AL, args[0], args[1]);

        break;



    default:

        tcg_abort();

    }

}
