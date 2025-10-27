static void tcg_out_op (TCGContext *s, TCGOpcode opc, const TCGArg *args,

                        const int *const_args)

{

    int c;



    switch (opc) {

    case INDEX_op_exit_tb:

        tcg_out_movi (s, TCG_TYPE_I64, TCG_REG_R3, args[0]);

        tcg_out_b (s, 0, (tcg_target_long) tb_ret_addr);

        break;

    case INDEX_op_goto_tb:

        if (s->tb_jmp_offset) {

            /* direct jump method */



            s->tb_jmp_offset[args[0]] = s->code_ptr - s->code_buf;

            s->code_ptr += 28;

        }

        else {

            tcg_abort ();

        }

        s->tb_next_offset[args[0]] = s->code_ptr - s->code_buf;

        break;

    case INDEX_op_br:

        {

            TCGLabel *l = &s->labels[args[0]];



            if (l->has_value) {

                tcg_out_b (s, 0, l->u.value);

            }

            else {

                uint32_t val = *(uint32_t *) s->code_ptr;



                /* Thanks to Andrzej Zaborowski */

                tcg_out32 (s, B | (val & 0x3fffffc));

                tcg_out_reloc (s, s->code_ptr - 4, R_PPC_REL24, args[0], 0);

            }

        }

        break;

    case INDEX_op_call:

        tcg_out_call (s, args[0], const_args[0]);

        break;

    case INDEX_op_jmp:

        if (const_args[0]) {

            tcg_out_b (s, 0, args[0]);

        }

        else {

            tcg_out32 (s, MTSPR | RS (args[0]) | CTR);

            tcg_out32 (s, BCCTR | BO_ALWAYS);

        }

        break;

    case INDEX_op_movi_i32:

        tcg_out_movi (s, TCG_TYPE_I32, args[0], args[1]);

        break;

    case INDEX_op_movi_i64:

        tcg_out_movi (s, TCG_TYPE_I64, args[0], args[1]);

        break;

    case INDEX_op_ld8u_i32:

    case INDEX_op_ld8u_i64:

        tcg_out_ldst (s, args[0], args[1], args[2], LBZ, LBZX);

        break;

    case INDEX_op_ld8s_i32:

    case INDEX_op_ld8s_i64:

        tcg_out_ldst (s, args[0], args[1], args[2], LBZ, LBZX);

        tcg_out32 (s, EXTSB | RS (args[0]) | RA (args[0]));

        break;

    case INDEX_op_ld16u_i32:

    case INDEX_op_ld16u_i64:

        tcg_out_ldst (s, args[0], args[1], args[2], LHZ, LHZX);

        break;

    case INDEX_op_ld16s_i32:

    case INDEX_op_ld16s_i64:

        tcg_out_ldst (s, args[0], args[1], args[2], LHA, LHAX);

        break;

    case INDEX_op_ld_i32:

    case INDEX_op_ld32u_i64:

        tcg_out_ldst (s, args[0], args[1], args[2], LWZ, LWZX);

        break;

    case INDEX_op_ld32s_i64:

        tcg_out_ldsta (s, args[0], args[1], args[2], LWA, LWAX);

        break;

    case INDEX_op_ld_i64:

        tcg_out_ldsta (s, args[0], args[1], args[2], LD, LDX);

        break;

    case INDEX_op_st8_i32:

    case INDEX_op_st8_i64:

        tcg_out_ldst (s, args[0], args[1], args[2], STB, STBX);

        break;

    case INDEX_op_st16_i32:

    case INDEX_op_st16_i64:

        tcg_out_ldst (s, args[0], args[1], args[2], STH, STHX);

        break;

    case INDEX_op_st_i32:

    case INDEX_op_st32_i64:

        tcg_out_ldst (s, args[0], args[1], args[2], STW, STWX);

        break;

    case INDEX_op_st_i64:

        tcg_out_ldsta (s, args[0], args[1], args[2], STD, STDX);

        break;



    case INDEX_op_add_i32:

        if (const_args[2])

            ppc_addi32 (s, args[0], args[1], args[2]);

        else

            tcg_out32 (s, ADD | TAB (args[0], args[1], args[2]));

        break;

    case INDEX_op_sub_i32:

        if (const_args[2])

            ppc_addi32 (s, args[0], args[1], -args[2]);

        else

            tcg_out32 (s, SUBF | TAB (args[0], args[2], args[1]));

        break;



    case INDEX_op_and_i64:

    case INDEX_op_and_i32:

        if (const_args[2]) {

            if ((args[2] & 0xffff) == args[2])

                tcg_out32 (s, ANDI | RS (args[1]) | RA (args[0]) | args[2]);

            else if ((args[2] & 0xffff0000) == args[2])

                tcg_out32 (s, ANDIS | RS (args[1]) | RA (args[0])

                           | ((args[2] >> 16) & 0xffff));

            else {

                tcg_out_movi (s, (opc == INDEX_op_and_i32

                                  ? TCG_TYPE_I32

                                  : TCG_TYPE_I64),

                              0, args[2]);

                tcg_out32 (s, AND | SAB (args[1], args[0], 0));

            }

        }

        else

            tcg_out32 (s, AND | SAB (args[1], args[0], args[2]));

        break;

    case INDEX_op_or_i64:

    case INDEX_op_or_i32:

        if (const_args[2]) {

            if (args[2] & 0xffff) {

                tcg_out32 (s, ORI | RS (args[1]) | RA (args[0])

                           | (args[2] & 0xffff));

                if (args[2] >> 16)

                    tcg_out32 (s, ORIS | RS (args[0])  | RA (args[0])

                               | ((args[2] >> 16) & 0xffff));

            }

            else {

                tcg_out32 (s, ORIS | RS (args[1])  | RA (args[0])

                           | ((args[2] >> 16) & 0xffff));

            }

        }

        else

            tcg_out32 (s, OR | SAB (args[1], args[0], args[2]));

        break;

    case INDEX_op_xor_i64:

    case INDEX_op_xor_i32:

        if (const_args[2]) {

            if ((args[2] & 0xffff) == args[2])

                tcg_out32 (s, XORI | RS (args[1])  | RA (args[0])

                           | (args[2] & 0xffff));

            else if ((args[2] & 0xffff0000) == args[2])

                tcg_out32 (s, XORIS | RS (args[1])  | RA (args[0])

                           | ((args[2] >> 16) & 0xffff));

            else {

                tcg_out_movi (s, (opc == INDEX_op_and_i32

                                  ? TCG_TYPE_I32

                                  : TCG_TYPE_I64),

                              0, args[2]);

                tcg_out32 (s, XOR | SAB (args[1], args[0], 0));

            }

        }

        else

            tcg_out32 (s, XOR | SAB (args[1], args[0], args[2]));

        break;



    case INDEX_op_mul_i32:

        if (const_args[2]) {

            if (args[2] == (int16_t) args[2])

                tcg_out32 (s, MULLI | RT (args[0]) | RA (args[1])

                           | (args[2] & 0xffff));

            else {

                tcg_out_movi (s, TCG_TYPE_I32, 0, args[2]);

                tcg_out32 (s, MULLW | TAB (args[0], args[1], 0));

            }

        }

        else

            tcg_out32 (s, MULLW | TAB (args[0], args[1], args[2]));

        break;



    case INDEX_op_div_i32:

        tcg_out32 (s, DIVW | TAB (args[0], args[1], args[2]));

        break;



    case INDEX_op_divu_i32:

        tcg_out32 (s, DIVWU | TAB (args[0], args[1], args[2]));

        break;



    case INDEX_op_rem_i32:

        tcg_out32 (s, DIVW | TAB (0, args[1], args[2]));

        tcg_out32 (s, MULLW | TAB (0, 0, args[2]));

        tcg_out32 (s, SUBF | TAB (args[0], 0, args[1]));

        break;



    case INDEX_op_remu_i32:

        tcg_out32 (s, DIVWU | TAB (0, args[1], args[2]));

        tcg_out32 (s, MULLW | TAB (0, 0, args[2]));

        tcg_out32 (s, SUBF | TAB (args[0], 0, args[1]));

        break;



    case INDEX_op_shl_i32:

        if (const_args[2]) {

            tcg_out32 (s, (RLWINM

                           | RA (args[0])

                           | RS (args[1])

                           | SH (args[2])

                           | MB (0)

                           | ME (31 - args[2])

                           )

                );

        }

        else

            tcg_out32 (s, SLW | SAB (args[1], args[0], args[2]));

        break;

    case INDEX_op_shr_i32:

        if (const_args[2]) {

            tcg_out32 (s, (RLWINM

                           | RA (args[0])

                           | RS (args[1])

                           | SH (32 - args[2])

                           | MB (args[2])

                           | ME (31)

                           )

                );

        }

        else

            tcg_out32 (s, SRW | SAB (args[1], args[0], args[2]));

        break;

    case INDEX_op_sar_i32:

        if (const_args[2])

            tcg_out32 (s, SRAWI | RS (args[1]) | RA (args[0]) | SH (args[2]));

        else

            tcg_out32 (s, SRAW | SAB (args[1], args[0], args[2]));

        break;



    case INDEX_op_brcond_i32:

        tcg_out_brcond (s, args[2], args[0], args[1], const_args[1], args[3], 0);

        break;



    case INDEX_op_brcond_i64:

        tcg_out_brcond (s, args[2], args[0], args[1], const_args[1], args[3], 1);

        break;



    case INDEX_op_neg_i32:

    case INDEX_op_neg_i64:

        tcg_out32 (s, NEG | RT (args[0]) | RA (args[1]));

        break;



    case INDEX_op_not_i32:

    case INDEX_op_not_i64:

        tcg_out32 (s, NOR | SAB (args[1], args[0], args[1]));

        break;



    case INDEX_op_add_i64:

        if (const_args[2])

            ppc_addi64 (s, args[0], args[1], args[2]);

        else

            tcg_out32 (s, ADD | TAB (args[0], args[1], args[2]));

        break;

    case INDEX_op_sub_i64:

        if (const_args[2])

            ppc_addi64 (s, args[0], args[1], -args[2]);

        else

            tcg_out32 (s, SUBF | TAB (args[0], args[2], args[1]));

        break;



    case INDEX_op_shl_i64:

        if (const_args[2])

            tcg_out_rld (s, RLDICR, args[0], args[1], args[2], 63 - args[2]);

        else

            tcg_out32 (s, SLD | SAB (args[1], args[0], args[2]));

        break;

    case INDEX_op_shr_i64:

        if (const_args[2])

            tcg_out_rld (s, RLDICL, args[0], args[1], 64 - args[2], args[2]);

        else

            tcg_out32 (s, SRD | SAB (args[1], args[0], args[2]));

        break;

    case INDEX_op_sar_i64:

        if (const_args[2]) {

            int sh = SH (args[2] & 0x1f) | (((args[2] >> 5) & 1) << 1);

            tcg_out32 (s, SRADI | RA (args[0]) | RS (args[1]) | sh);

        }

        else

            tcg_out32 (s, SRAD | SAB (args[1], args[0], args[2]));

        break;



    case INDEX_op_mul_i64:

        tcg_out32 (s, MULLD | TAB (args[0], args[1], args[2]));

        break;

    case INDEX_op_div_i64:

        tcg_out32 (s, DIVD | TAB (args[0], args[1], args[2]));

        break;

    case INDEX_op_divu_i64:

        tcg_out32 (s, DIVDU | TAB (args[0], args[1], args[2]));

        break;

    case INDEX_op_rem_i64:

        tcg_out32 (s, DIVD | TAB (0, args[1], args[2]));

        tcg_out32 (s, MULLD | TAB (0, 0, args[2]));

        tcg_out32 (s, SUBF | TAB (args[0], 0, args[1]));

        break;

    case INDEX_op_remu_i64:

        tcg_out32 (s, DIVDU | TAB (0, args[1], args[2]));

        tcg_out32 (s, MULLD | TAB (0, 0, args[2]));

        tcg_out32 (s, SUBF | TAB (args[0], 0, args[1]));

        break;



    case INDEX_op_qemu_ld8u:

        tcg_out_qemu_ld (s, args, 0);

        break;

    case INDEX_op_qemu_ld8s:

        tcg_out_qemu_ld (s, args, 0 | 4);

        break;

    case INDEX_op_qemu_ld16u:

        tcg_out_qemu_ld (s, args, 1);

        break;

    case INDEX_op_qemu_ld16s:

        tcg_out_qemu_ld (s, args, 1 | 4);

        break;

    case INDEX_op_qemu_ld32:

    case INDEX_op_qemu_ld32u:

        tcg_out_qemu_ld (s, args, 2);

        break;

    case INDEX_op_qemu_ld32s:

        tcg_out_qemu_ld (s, args, 2 | 4);

        break;

    case INDEX_op_qemu_ld64:

        tcg_out_qemu_ld (s, args, 3);

        break;

    case INDEX_op_qemu_st8:

        tcg_out_qemu_st (s, args, 0);

        break;

    case INDEX_op_qemu_st16:

        tcg_out_qemu_st (s, args, 1);

        break;

    case INDEX_op_qemu_st32:

        tcg_out_qemu_st (s, args, 2);

        break;

    case INDEX_op_qemu_st64:

        tcg_out_qemu_st (s, args, 3);

        break;



    case INDEX_op_ext8s_i32:

    case INDEX_op_ext8s_i64:

        c = EXTSB;

        goto gen_ext;

    case INDEX_op_ext16s_i32:

    case INDEX_op_ext16s_i64:

        c = EXTSH;

        goto gen_ext;

    case INDEX_op_ext32s_i64:

        c = EXTSW;

        goto gen_ext;

    gen_ext:

        tcg_out32 (s, c | RS (args[1]) | RA (args[0]));

        break;



    case INDEX_op_ext32u_i64:

        tcg_out_rld (s, RLDICR, args[0], args[1], 0, 32);

        break;



    case INDEX_op_setcond_i32:

        tcg_out_setcond (s, TCG_TYPE_I32, args[3], args[0], args[1], args[2],

                         const_args[2]);

        break;

    case INDEX_op_setcond_i64:

        tcg_out_setcond (s, TCG_TYPE_I64, args[3], args[0], args[1], args[2],

                         const_args[2]);

        break;



    default:

        tcg_dump_ops (s, stderr);

        tcg_abort ();

    }

}
