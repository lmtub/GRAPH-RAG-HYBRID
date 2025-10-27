static inline void tcg_out_op(TCGContext *s, TCGOpcode opc,

                              const TCGArg *args, const int *const_args)

{

    int c, vexop, rexw = 0;



#if TCG_TARGET_REG_BITS == 64

# define OP_32_64(x) \

        case glue(glue(INDEX_op_, x), _i64): \

            rexw = P_REXW; /* FALLTHRU */    \

        case glue(glue(INDEX_op_, x), _i32)

#else

# define OP_32_64(x) \

        case glue(glue(INDEX_op_, x), _i32)

#endif



    switch(opc) {

    case INDEX_op_exit_tb:

        tcg_out_movi(s, TCG_TYPE_PTR, TCG_REG_EAX, args[0]);

        tcg_out_jmp(s, tb_ret_addr);

        break;

    case INDEX_op_goto_tb:

        if (s->tb_jmp_offset) {

            /* direct jump method */

            tcg_out8(s, OPC_JMP_long); /* jmp im */

            s->tb_jmp_offset[args[0]] = tcg_current_code_size(s);

            tcg_out32(s, 0);

        } else {

            /* indirect jump method */

            tcg_out_modrm_offset(s, OPC_GRP5, EXT5_JMPN_Ev, -1,

                                 (intptr_t)(s->tb_next + args[0]));

        }

        s->tb_next_offset[args[0]] = tcg_current_code_size(s);

        break;

    case INDEX_op_br:

        tcg_out_jxx(s, JCC_JMP, args[0], 0);

        break;

    OP_32_64(ld8u):

        /* Note that we can ignore REXW for the zero-extend to 64-bit.  */

        tcg_out_modrm_offset(s, OPC_MOVZBL, args[0], args[1], args[2]);

        break;

    OP_32_64(ld8s):

        tcg_out_modrm_offset(s, OPC_MOVSBL + rexw, args[0], args[1], args[2]);

        break;

    OP_32_64(ld16u):

        /* Note that we can ignore REXW for the zero-extend to 64-bit.  */

        tcg_out_modrm_offset(s, OPC_MOVZWL, args[0], args[1], args[2]);

        break;

    OP_32_64(ld16s):

        tcg_out_modrm_offset(s, OPC_MOVSWL + rexw, args[0], args[1], args[2]);

        break;

#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_ld32u_i64:

#endif

    case INDEX_op_ld_i32:

        tcg_out_ld(s, TCG_TYPE_I32, args[0], args[1], args[2]);

        break;



    OP_32_64(st8):

        if (const_args[0]) {

            tcg_out_modrm_offset(s, OPC_MOVB_EvIz,

                                 0, args[1], args[2]);

            tcg_out8(s, args[0]);

        } else {

            tcg_out_modrm_offset(s, OPC_MOVB_EvGv | P_REXB_R,

                                 args[0], args[1], args[2]);

        }

        break;

    OP_32_64(st16):

        if (const_args[0]) {

            tcg_out_modrm_offset(s, OPC_MOVL_EvIz | P_DATA16,

                                 0, args[1], args[2]);

            tcg_out16(s, args[0]);

        } else {

            tcg_out_modrm_offset(s, OPC_MOVL_EvGv | P_DATA16,

                                 args[0], args[1], args[2]);

        }

        break;

#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_st32_i64:

#endif

    case INDEX_op_st_i32:

        if (const_args[0]) {

            tcg_out_modrm_offset(s, OPC_MOVL_EvIz, 0, args[1], args[2]);

            tcg_out32(s, args[0]);

        } else {

            tcg_out_st(s, TCG_TYPE_I32, args[0], args[1], args[2]);

        }

        break;



    OP_32_64(add):

        /* For 3-operand addition, use LEA.  */

        if (args[0] != args[1]) {

            TCGArg a0 = args[0], a1 = args[1], a2 = args[2], c3 = 0;



            if (const_args[2]) {

                c3 = a2, a2 = -1;

            } else if (a0 == a2) {

                /* Watch out for dest = src + dest, since we've removed

                   the matching constraint on the add.  */

                tgen_arithr(s, ARITH_ADD + rexw, a0, a1);

                break;

            }



            tcg_out_modrm_sib_offset(s, OPC_LEA + rexw, a0, a1, a2, 0, c3);

            break;

        }

        c = ARITH_ADD;

        goto gen_arith;

    OP_32_64(sub):

        c = ARITH_SUB;

        goto gen_arith;

    OP_32_64(and):

        c = ARITH_AND;

        goto gen_arith;

    OP_32_64(or):

        c = ARITH_OR;

        goto gen_arith;

    OP_32_64(xor):

        c = ARITH_XOR;

        goto gen_arith;

    gen_arith:

        if (const_args[2]) {

            tgen_arithi(s, c + rexw, args[0], args[2], 0);

        } else {

            tgen_arithr(s, c + rexw, args[0], args[2]);

        }

        break;



    OP_32_64(andc):

        if (const_args[2]) {

            tcg_out_mov(s, rexw ? TCG_TYPE_I64 : TCG_TYPE_I32,

                        args[0], args[1]);

            tgen_arithi(s, ARITH_AND + rexw, args[0], ~args[2], 0);

        } else {

            tcg_out_vex_modrm(s, OPC_ANDN + rexw, args[0], args[2], args[1]);

        }

        break;



    OP_32_64(mul):

        if (const_args[2]) {

            int32_t val;

            val = args[2];

            if (val == (int8_t)val) {

                tcg_out_modrm(s, OPC_IMUL_GvEvIb + rexw, args[0], args[0]);

                tcg_out8(s, val);

            } else {

                tcg_out_modrm(s, OPC_IMUL_GvEvIz + rexw, args[0], args[0]);

                tcg_out32(s, val);

            }

        } else {

            tcg_out_modrm(s, OPC_IMUL_GvEv + rexw, args[0], args[2]);

        }

        break;



    OP_32_64(div2):

        tcg_out_modrm(s, OPC_GRP3_Ev + rexw, EXT3_IDIV, args[4]);

        break;

    OP_32_64(divu2):

        tcg_out_modrm(s, OPC_GRP3_Ev + rexw, EXT3_DIV, args[4]);

        break;



    OP_32_64(shl):

        c = SHIFT_SHL;

        vexop = OPC_SHLX;

        goto gen_shift_maybe_vex;

    OP_32_64(shr):

        c = SHIFT_SHR;

        vexop = OPC_SHRX;

        goto gen_shift_maybe_vex;

    OP_32_64(sar):

        c = SHIFT_SAR;

        vexop = OPC_SARX;

        goto gen_shift_maybe_vex;

    OP_32_64(rotl):

        c = SHIFT_ROL;

        goto gen_shift;

    OP_32_64(rotr):

        c = SHIFT_ROR;

        goto gen_shift;

    gen_shift_maybe_vex:

        if (have_bmi2 && !const_args[2]) {

            tcg_out_vex_modrm(s, vexop + rexw, args[0], args[2], args[1]);

            break;

        }

        /* FALLTHRU */

    gen_shift:

        if (const_args[2]) {

            tcg_out_shifti(s, c + rexw, args[0], args[2]);

        } else {

            tcg_out_modrm(s, OPC_SHIFT_cl + rexw, c, args[0]);

        }

        break;



    case INDEX_op_brcond_i32:

        tcg_out_brcond32(s, args[2], args[0], args[1], const_args[1],

                         args[3], 0);

        break;

    case INDEX_op_setcond_i32:

        tcg_out_setcond32(s, args[3], args[0], args[1],

                          args[2], const_args[2]);

        break;

    case INDEX_op_movcond_i32:

        tcg_out_movcond32(s, args[5], args[0], args[1],

                          args[2], const_args[2], args[3]);

        break;



    OP_32_64(bswap16):

        tcg_out_rolw_8(s, args[0]);

        break;

    OP_32_64(bswap32):

        tcg_out_bswap32(s, args[0]);

        break;



    OP_32_64(neg):

        tcg_out_modrm(s, OPC_GRP3_Ev + rexw, EXT3_NEG, args[0]);

        break;

    OP_32_64(not):

        tcg_out_modrm(s, OPC_GRP3_Ev + rexw, EXT3_NOT, args[0]);

        break;



    OP_32_64(ext8s):

        tcg_out_ext8s(s, args[0], args[1], rexw);

        break;

    OP_32_64(ext16s):

        tcg_out_ext16s(s, args[0], args[1], rexw);

        break;

    OP_32_64(ext8u):

        tcg_out_ext8u(s, args[0], args[1]);

        break;

    OP_32_64(ext16u):

        tcg_out_ext16u(s, args[0], args[1]);

        break;



    case INDEX_op_qemu_ld_i32:

        tcg_out_qemu_ld(s, args, 0);

        break;

    case INDEX_op_qemu_ld_i64:

        tcg_out_qemu_ld(s, args, 1);

        break;

    case INDEX_op_qemu_st_i32:

        tcg_out_qemu_st(s, args, 0);

        break;

    case INDEX_op_qemu_st_i64:

        tcg_out_qemu_st(s, args, 1);

        break;



    OP_32_64(mulu2):

        tcg_out_modrm(s, OPC_GRP3_Ev + rexw, EXT3_MUL, args[3]);

        break;

    OP_32_64(muls2):

        tcg_out_modrm(s, OPC_GRP3_Ev + rexw, EXT3_IMUL, args[3]);

        break;

    OP_32_64(add2):

        if (const_args[4]) {

            tgen_arithi(s, ARITH_ADD + rexw, args[0], args[4], 1);

        } else {

            tgen_arithr(s, ARITH_ADD + rexw, args[0], args[4]);

        }

        if (const_args[5]) {

            tgen_arithi(s, ARITH_ADC + rexw, args[1], args[5], 1);

        } else {

            tgen_arithr(s, ARITH_ADC + rexw, args[1], args[5]);

        }

        break;

    OP_32_64(sub2):

        if (const_args[4]) {

            tgen_arithi(s, ARITH_SUB + rexw, args[0], args[4], 1);

        } else {

            tgen_arithr(s, ARITH_SUB + rexw, args[0], args[4]);

        }

        if (const_args[5]) {

            tgen_arithi(s, ARITH_SBB + rexw, args[1], args[5], 1);

        } else {

            tgen_arithr(s, ARITH_SBB + rexw, args[1], args[5]);

        }

        break;



#if TCG_TARGET_REG_BITS == 32

    case INDEX_op_brcond2_i32:

        tcg_out_brcond2(s, args, const_args, 0);

        break;

    case INDEX_op_setcond2_i32:

        tcg_out_setcond2(s, args, const_args);

        break;

#else /* TCG_TARGET_REG_BITS == 64 */

    case INDEX_op_ld32s_i64:

        tcg_out_modrm_offset(s, OPC_MOVSLQ, args[0], args[1], args[2]);

        break;

    case INDEX_op_ld_i64:

        tcg_out_ld(s, TCG_TYPE_I64, args[0], args[1], args[2]);

        break;

    case INDEX_op_st_i64:

        if (const_args[0]) {

            tcg_out_modrm_offset(s, OPC_MOVL_EvIz | P_REXW,

                                 0, args[1], args[2]);

            tcg_out32(s, args[0]);

        } else {

            tcg_out_st(s, TCG_TYPE_I64, args[0], args[1], args[2]);

        }

        break;



    case INDEX_op_brcond_i64:

        tcg_out_brcond64(s, args[2], args[0], args[1], const_args[1],

                         args[3], 0);

        break;

    case INDEX_op_setcond_i64:

        tcg_out_setcond64(s, args[3], args[0], args[1],

                          args[2], const_args[2]);

        break;

    case INDEX_op_movcond_i64:

        tcg_out_movcond64(s, args[5], args[0], args[1],

                          args[2], const_args[2], args[3]);

        break;



    case INDEX_op_bswap64_i64:

        tcg_out_bswap64(s, args[0]);

        break;

    case INDEX_op_ext32u_i64:

        tcg_out_ext32u(s, args[0], args[1]);

        break;

    case INDEX_op_ext32s_i64:

        tcg_out_ext32s(s, args[0], args[1]);

        break;

#endif



    OP_32_64(deposit):

        if (args[3] == 0 && args[4] == 8) {

            /* load bits 0..7 */

            tcg_out_modrm(s, OPC_MOVB_EvGv | P_REXB_R | P_REXB_RM,

                          args[2], args[0]);

        } else if (args[3] == 8 && args[4] == 8) {

            /* load bits 8..15 */

            tcg_out_modrm(s, OPC_MOVB_EvGv, args[2], args[0] + 4);

        } else if (args[3] == 0 && args[4] == 16) {

            /* load bits 0..15 */

            tcg_out_modrm(s, OPC_MOVL_EvGv | P_DATA16, args[2], args[0]);

        } else {

            tcg_abort();

        }

        break;



    case INDEX_op_mov_i32:  /* Always emitted via tcg_out_mov.  */

    case INDEX_op_mov_i64:

    case INDEX_op_movi_i32: /* Always emitted via tcg_out_movi.  */

    case INDEX_op_movi_i64:

    case INDEX_op_call:     /* Always emitted via tcg_out_call.  */

    default:

        tcg_abort();

    }



#undef OP_32_64

}
