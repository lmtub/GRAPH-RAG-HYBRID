static int disas_neon_data_insn(CPUState * env, DisasContext *s, uint32_t insn)

{

    int op;

    int q;

    int rd, rn, rm;

    int size;

    int shift;

    int pass;

    int count;

    int pairwise;

    int u;

    uint32_t imm, mask;

    TCGv tmp, tmp2, tmp3, tmp4, tmp5;

    TCGv_i64 tmp64;



    if (!s->vfp_enabled)

      return 1;

    q = (insn & (1 << 6)) != 0;

    u = (insn >> 24) & 1;

    VFP_DREG_D(rd, insn);

    VFP_DREG_N(rn, insn);

    VFP_DREG_M(rm, insn);

    size = (insn >> 20) & 3;

    if ((insn & (1 << 23)) == 0) {

        /* Three register same length.  */

        op = ((insn >> 7) & 0x1e) | ((insn >> 4) & 1);

        /* Catch invalid op and bad size combinations: UNDEF */

        if ((neon_3r_sizes[op] & (1 << size)) == 0) {

            return 1;

        }

        /* All insns of this form UNDEF for either this condition or the

         * superset of cases "Q==1"; we catch the latter later.

         */

        if (q && ((rd | rn | rm) & 1)) {

            return 1;

        }

        if (size == 3 && op != NEON_3R_LOGIC) {

            /* 64-bit element instructions. */

            for (pass = 0; pass < (q ? 2 : 1); pass++) {

                neon_load_reg64(cpu_V0, rn + pass);

                neon_load_reg64(cpu_V1, rm + pass);

                switch (op) {

                case NEON_3R_VQADD:

                    if (u) {

                        gen_helper_neon_qadd_u64(cpu_V0, cpu_V0, cpu_V1);

                    } else {

                        gen_helper_neon_qadd_s64(cpu_V0, cpu_V0, cpu_V1);

                    }

                    break;

                case NEON_3R_VQSUB:

                    if (u) {

                        gen_helper_neon_qsub_u64(cpu_V0, cpu_V0, cpu_V1);

                    } else {

                        gen_helper_neon_qsub_s64(cpu_V0, cpu_V0, cpu_V1);

                    }

                    break;

                case NEON_3R_VSHL:

                    if (u) {

                        gen_helper_neon_shl_u64(cpu_V0, cpu_V1, cpu_V0);

                    } else {

                        gen_helper_neon_shl_s64(cpu_V0, cpu_V1, cpu_V0);

                    }

                    break;

                case NEON_3R_VQSHL:

                    if (u) {

                        gen_helper_neon_qshl_u64(cpu_V0, cpu_V1, cpu_V0);

                    } else {

                        gen_helper_neon_qshl_s64(cpu_V0, cpu_V1, cpu_V0);

                    }

                    break;

                case NEON_3R_VRSHL:

                    if (u) {

                        gen_helper_neon_rshl_u64(cpu_V0, cpu_V1, cpu_V0);

                    } else {

                        gen_helper_neon_rshl_s64(cpu_V0, cpu_V1, cpu_V0);

                    }

                    break;

                case NEON_3R_VQRSHL:

                    if (u) {

                        gen_helper_neon_qrshl_u64(cpu_V0, cpu_V1, cpu_V0);

                    } else {

                        gen_helper_neon_qrshl_s64(cpu_V0, cpu_V1, cpu_V0);

                    }

                    break;

                case NEON_3R_VADD_VSUB:

                    if (u) {

                        tcg_gen_sub_i64(CPU_V001);

                    } else {

                        tcg_gen_add_i64(CPU_V001);

                    }

                    break;

                default:

                    abort();

                }

                neon_store_reg64(cpu_V0, rd + pass);

            }

            return 0;

        }

        pairwise = 0;

        switch (op) {

        case NEON_3R_VSHL:

        case NEON_3R_VQSHL:

        case NEON_3R_VRSHL:

        case NEON_3R_VQRSHL:

            {

                int rtmp;

                /* Shift instruction operands are reversed.  */

                rtmp = rn;

                rn = rm;

                rm = rtmp;

            }

            break;

        case NEON_3R_VPADD:

            if (u) {

                return 1;

            }

            /* Fall through */

        case NEON_3R_VPMAX:

        case NEON_3R_VPMIN:

            pairwise = 1;

            break;

        case NEON_3R_FLOAT_ARITH:

            pairwise = (u && size < 2); /* if VPADD (float) */

            break;

        case NEON_3R_FLOAT_MINMAX:

            pairwise = u; /* if VPMIN/VPMAX (float) */

            break;

        case NEON_3R_FLOAT_CMP:

            if (!u && size) {

                /* no encoding for U=0 C=1x */

                return 1;

            }

            break;

        case NEON_3R_FLOAT_ACMP:

            if (!u) {

                return 1;

            }

            break;

        case NEON_3R_VRECPS_VRSQRTS:

            if (u) {

                return 1;

            }

            break;

        case NEON_3R_VMUL:

            if (u && (size != 0)) {

                /* UNDEF on invalid size for polynomial subcase */

                return 1;

            }

            break;

        default:

            break;

        }



        if (pairwise && q) {

            /* All the pairwise insns UNDEF if Q is set */

            return 1;

        }



        for (pass = 0; pass < (q ? 4 : 2); pass++) {



        if (pairwise) {

            /* Pairwise.  */

            if (pass < 1) {

                tmp = neon_load_reg(rn, 0);

                tmp2 = neon_load_reg(rn, 1);

            } else {

                tmp = neon_load_reg(rm, 0);

                tmp2 = neon_load_reg(rm, 1);

            }

        } else {

            /* Elementwise.  */

            tmp = neon_load_reg(rn, pass);

            tmp2 = neon_load_reg(rm, pass);

        }

        switch (op) {

        case NEON_3R_VHADD:

            GEN_NEON_INTEGER_OP(hadd);

            break;

        case NEON_3R_VQADD:

            GEN_NEON_INTEGER_OP(qadd);

            break;

        case NEON_3R_VRHADD:

            GEN_NEON_INTEGER_OP(rhadd);

            break;

        case NEON_3R_LOGIC: /* Logic ops.  */

            switch ((u << 2) | size) {

            case 0: /* VAND */

                tcg_gen_and_i32(tmp, tmp, tmp2);

                break;

            case 1: /* BIC */

                tcg_gen_andc_i32(tmp, tmp, tmp2);

                break;

            case 2: /* VORR */

                tcg_gen_or_i32(tmp, tmp, tmp2);

                break;

            case 3: /* VORN */

                tcg_gen_orc_i32(tmp, tmp, tmp2);

                break;

            case 4: /* VEOR */

                tcg_gen_xor_i32(tmp, tmp, tmp2);

                break;

            case 5: /* VBSL */

                tmp3 = neon_load_reg(rd, pass);

                gen_neon_bsl(tmp, tmp, tmp2, tmp3);

                tcg_temp_free_i32(tmp3);

                break;

            case 6: /* VBIT */

                tmp3 = neon_load_reg(rd, pass);

                gen_neon_bsl(tmp, tmp, tmp3, tmp2);

                tcg_temp_free_i32(tmp3);

                break;

            case 7: /* VBIF */

                tmp3 = neon_load_reg(rd, pass);

                gen_neon_bsl(tmp, tmp3, tmp, tmp2);

                tcg_temp_free_i32(tmp3);

                break;

            }

            break;

        case NEON_3R_VHSUB:

            GEN_NEON_INTEGER_OP(hsub);

            break;

        case NEON_3R_VQSUB:

            GEN_NEON_INTEGER_OP(qsub);

            break;

        case NEON_3R_VCGT:

            GEN_NEON_INTEGER_OP(cgt);

            break;

        case NEON_3R_VCGE:

            GEN_NEON_INTEGER_OP(cge);

            break;

        case NEON_3R_VSHL:

            GEN_NEON_INTEGER_OP(shl);

            break;

        case NEON_3R_VQSHL:

            GEN_NEON_INTEGER_OP(qshl);

            break;

        case NEON_3R_VRSHL:

            GEN_NEON_INTEGER_OP(rshl);

            break;

        case NEON_3R_VQRSHL:

            GEN_NEON_INTEGER_OP(qrshl);

            break;

        case NEON_3R_VMAX:

            GEN_NEON_INTEGER_OP(max);

            break;

        case NEON_3R_VMIN:

            GEN_NEON_INTEGER_OP(min);

            break;

        case NEON_3R_VABD:

            GEN_NEON_INTEGER_OP(abd);

            break;

        case NEON_3R_VABA:

            GEN_NEON_INTEGER_OP(abd);

            tcg_temp_free_i32(tmp2);

            tmp2 = neon_load_reg(rd, pass);

            gen_neon_add(size, tmp, tmp2);

            break;

        case NEON_3R_VADD_VSUB:

            if (!u) { /* VADD */

                gen_neon_add(size, tmp, tmp2);

            } else { /* VSUB */

                switch (size) {

                case 0: gen_helper_neon_sub_u8(tmp, tmp, tmp2); break;

                case 1: gen_helper_neon_sub_u16(tmp, tmp, tmp2); break;

                case 2: tcg_gen_sub_i32(tmp, tmp, tmp2); break;

                default: abort();

                }

            }

            break;

        case NEON_3R_VTST_VCEQ:

            if (!u) { /* VTST */

                switch (size) {

                case 0: gen_helper_neon_tst_u8(tmp, tmp, tmp2); break;

                case 1: gen_helper_neon_tst_u16(tmp, tmp, tmp2); break;

                case 2: gen_helper_neon_tst_u32(tmp, tmp, tmp2); break;

                default: abort();

                }

            } else { /* VCEQ */

                switch (size) {

                case 0: gen_helper_neon_ceq_u8(tmp, tmp, tmp2); break;

                case 1: gen_helper_neon_ceq_u16(tmp, tmp, tmp2); break;

                case 2: gen_helper_neon_ceq_u32(tmp, tmp, tmp2); break;

                default: abort();

                }

            }

            break;

        case NEON_3R_VML: /* VMLA, VMLAL, VMLS,VMLSL */

            switch (size) {

            case 0: gen_helper_neon_mul_u8(tmp, tmp, tmp2); break;

            case 1: gen_helper_neon_mul_u16(tmp, tmp, tmp2); break;

            case 2: tcg_gen_mul_i32(tmp, tmp, tmp2); break;

            default: abort();

            }

            tcg_temp_free_i32(tmp2);

            tmp2 = neon_load_reg(rd, pass);

            if (u) { /* VMLS */

                gen_neon_rsb(size, tmp, tmp2);

            } else { /* VMLA */

                gen_neon_add(size, tmp, tmp2);

            }

            break;

        case NEON_3R_VMUL:

            if (u) { /* polynomial */

                gen_helper_neon_mul_p8(tmp, tmp, tmp2);

            } else { /* Integer */

                switch (size) {

                case 0: gen_helper_neon_mul_u8(tmp, tmp, tmp2); break;

                case 1: gen_helper_neon_mul_u16(tmp, tmp, tmp2); break;

                case 2: tcg_gen_mul_i32(tmp, tmp, tmp2); break;

                default: abort();

                }

            }

            break;

        case NEON_3R_VPMAX:

            GEN_NEON_INTEGER_OP(pmax);

            break;

        case NEON_3R_VPMIN:

            GEN_NEON_INTEGER_OP(pmin);

            break;

        case NEON_3R_VQDMULH_VQRDMULH: /* Multiply high.  */

            if (!u) { /* VQDMULH */

                switch (size) {

                case 1: gen_helper_neon_qdmulh_s16(tmp, tmp, tmp2); break;

                case 2: gen_helper_neon_qdmulh_s32(tmp, tmp, tmp2); break;

                default: abort();

                }

            } else { /* VQRDMULH */

                switch (size) {

                case 1: gen_helper_neon_qrdmulh_s16(tmp, tmp, tmp2); break;

                case 2: gen_helper_neon_qrdmulh_s32(tmp, tmp, tmp2); break;

                default: abort();

                }

            }

            break;

        case NEON_3R_VPADD:

            switch (size) {

            case 0: gen_helper_neon_padd_u8(tmp, tmp, tmp2); break;

            case 1: gen_helper_neon_padd_u16(tmp, tmp, tmp2); break;

            case 2: tcg_gen_add_i32(tmp, tmp, tmp2); break;

            default: abort();

            }

            break;

        case NEON_3R_FLOAT_ARITH: /* Floating point arithmetic. */

            switch ((u << 2) | size) {

            case 0: /* VADD */

                gen_helper_neon_add_f32(tmp, tmp, tmp2);

                break;

            case 2: /* VSUB */

                gen_helper_neon_sub_f32(tmp, tmp, tmp2);

                break;

            case 4: /* VPADD */

                gen_helper_neon_add_f32(tmp, tmp, tmp2);

                break;

            case 6: /* VABD */

                gen_helper_neon_abd_f32(tmp, tmp, tmp2);

                break;

            default:

                abort();

            }

            break;

        case NEON_3R_FLOAT_MULTIPLY:

            gen_helper_neon_mul_f32(tmp, tmp, tmp2);

            if (!u) {

                tcg_temp_free_i32(tmp2);

                tmp2 = neon_load_reg(rd, pass);

                if (size == 0) {

                    gen_helper_neon_add_f32(tmp, tmp, tmp2);

                } else {

                    gen_helper_neon_sub_f32(tmp, tmp2, tmp);

                }

            }

            break;

        case NEON_3R_FLOAT_CMP:

            if (!u) {

                gen_helper_neon_ceq_f32(tmp, tmp, tmp2);

            } else {

                if (size == 0)

                    gen_helper_neon_cge_f32(tmp, tmp, tmp2);

                else

                    gen_helper_neon_cgt_f32(tmp, tmp, tmp2);

            }

            break;

        case NEON_3R_FLOAT_ACMP:

            if (size == 0)

                gen_helper_neon_acge_f32(tmp, tmp, tmp2);

            else

                gen_helper_neon_acgt_f32(tmp, tmp, tmp2);

            break;

        case NEON_3R_FLOAT_MINMAX:

            if (size == 0)

                gen_helper_neon_max_f32(tmp, tmp, tmp2);

            else

                gen_helper_neon_min_f32(tmp, tmp, tmp2);

            break;

        case NEON_3R_VRECPS_VRSQRTS:

            if (size == 0)

                gen_helper_recps_f32(tmp, tmp, tmp2, cpu_env);

            else

                gen_helper_rsqrts_f32(tmp, tmp, tmp2, cpu_env);

            break;

        default:

            abort();

        }

        tcg_temp_free_i32(tmp2);



        /* Save the result.  For elementwise operations we can put it

           straight into the destination register.  For pairwise operations

           we have to be careful to avoid clobbering the source operands.  */

        if (pairwise && rd == rm) {

            neon_store_scratch(pass, tmp);

        } else {

            neon_store_reg(rd, pass, tmp);

        }



        } /* for pass */

        if (pairwise && rd == rm) {

            for (pass = 0; pass < (q ? 4 : 2); pass++) {

                tmp = neon_load_scratch(pass);

                neon_store_reg(rd, pass, tmp);

            }

        }

        /* End of 3 register same size operations.  */

    } else if (insn & (1 << 4)) {

        if ((insn & 0x00380080) != 0) {

            /* Two registers and shift.  */

            op = (insn >> 8) & 0xf;

            if (insn & (1 << 7)) {

                /* 64-bit shift. */

                if (op > 7) {

                    return 1;

                }

                size = 3;

            } else {

                size = 2;

                while ((insn & (1 << (size + 19))) == 0)

                    size--;

            }

            shift = (insn >> 16) & ((1 << (3 + size)) - 1);

            /* To avoid excessive dumplication of ops we implement shift

               by immediate using the variable shift operations.  */

            if (op < 8) {

                /* Shift by immediate:

                   VSHR, VSRA, VRSHR, VRSRA, VSRI, VSHL, VQSHL, VQSHLU.  */

                if (q && ((rd | rm) & 1)) {

                    return 1;

                }

                if (!u && (op == 4 || op == 6)) {

                    return 1;

                }

                /* Right shifts are encoded as N - shift, where N is the

                   element size in bits.  */

                if (op <= 4)

                    shift = shift - (1 << (size + 3));

                if (size == 3) {

                    count = q + 1;

                } else {

                    count = q ? 4: 2;

                }

                switch (size) {

                case 0:

                    imm = (uint8_t) shift;

                    imm |= imm << 8;

                    imm |= imm << 16;

                    break;

                case 1:

                    imm = (uint16_t) shift;

                    imm |= imm << 16;

                    break;

                case 2:

                case 3:

                    imm = shift;

                    break;

                default:

                    abort();

                }



                for (pass = 0; pass < count; pass++) {

                    if (size == 3) {

                        neon_load_reg64(cpu_V0, rm + pass);

                        tcg_gen_movi_i64(cpu_V1, imm);

                        switch (op) {

                        case 0:  /* VSHR */

                        case 1:  /* VSRA */

                            if (u)

                                gen_helper_neon_shl_u64(cpu_V0, cpu_V0, cpu_V1);

                            else

                                gen_helper_neon_shl_s64(cpu_V0, cpu_V0, cpu_V1);

                            break;

                        case 2: /* VRSHR */

                        case 3: /* VRSRA */

                            if (u)

                                gen_helper_neon_rshl_u64(cpu_V0, cpu_V0, cpu_V1);

                            else

                                gen_helper_neon_rshl_s64(cpu_V0, cpu_V0, cpu_V1);

                            break;

                        case 4: /* VSRI */

                        case 5: /* VSHL, VSLI */

                            gen_helper_neon_shl_u64(cpu_V0, cpu_V0, cpu_V1);

                            break;

                        case 6: /* VQSHLU */

                            gen_helper_neon_qshlu_s64(cpu_V0, cpu_V0, cpu_V1);

                            break;

                        case 7: /* VQSHL */

                            if (u) {

                                gen_helper_neon_qshl_u64(cpu_V0,

                                                         cpu_V0, cpu_V1);

                            } else {

                                gen_helper_neon_qshl_s64(cpu_V0,

                                                         cpu_V0, cpu_V1);

                            }

                            break;

                        }

                        if (op == 1 || op == 3) {

                            /* Accumulate.  */

                            neon_load_reg64(cpu_V1, rd + pass);

                            tcg_gen_add_i64(cpu_V0, cpu_V0, cpu_V1);

                        } else if (op == 4 || (op == 5 && u)) {

                            /* Insert */

                            neon_load_reg64(cpu_V1, rd + pass);

                            uint64_t mask;

                            if (shift < -63 || shift > 63) {

                                mask = 0;

                            } else {

                                if (op == 4) {

                                    mask = 0xffffffffffffffffull >> -shift;

                                } else {

                                    mask = 0xffffffffffffffffull << shift;

                                }

                            }

                            tcg_gen_andi_i64(cpu_V1, cpu_V1, ~mask);

                            tcg_gen_or_i64(cpu_V0, cpu_V0, cpu_V1);

                        }

                        neon_store_reg64(cpu_V0, rd + pass);

                    } else { /* size < 3 */

                        /* Operands in T0 and T1.  */

                        tmp = neon_load_reg(rm, pass);

                        tmp2 = tcg_temp_new_i32();

                        tcg_gen_movi_i32(tmp2, imm);

                        switch (op) {

                        case 0:  /* VSHR */

                        case 1:  /* VSRA */

                            GEN_NEON_INTEGER_OP(shl);

                            break;

                        case 2: /* VRSHR */

                        case 3: /* VRSRA */

                            GEN_NEON_INTEGER_OP(rshl);

                            break;

                        case 4: /* VSRI */

                        case 5: /* VSHL, VSLI */

                            switch (size) {

                            case 0: gen_helper_neon_shl_u8(tmp, tmp, tmp2); break;

                            case 1: gen_helper_neon_shl_u16(tmp, tmp, tmp2); break;

                            case 2: gen_helper_neon_shl_u32(tmp, tmp, tmp2); break;

                            default: abort();

                            }

                            break;

                        case 6: /* VQSHLU */

                            switch (size) {

                            case 0:

                                gen_helper_neon_qshlu_s8(tmp, tmp, tmp2);

                                break;

                            case 1:

                                gen_helper_neon_qshlu_s16(tmp, tmp, tmp2);

                                break;

                            case 2:

                                gen_helper_neon_qshlu_s32(tmp, tmp, tmp2);

                                break;

                            default:

                                abort();

                            }

                            break;

                        case 7: /* VQSHL */

                            GEN_NEON_INTEGER_OP(qshl);

                            break;

                        }

                        tcg_temp_free_i32(tmp2);



                        if (op == 1 || op == 3) {

                            /* Accumulate.  */

                            tmp2 = neon_load_reg(rd, pass);

                            gen_neon_add(size, tmp, tmp2);

                            tcg_temp_free_i32(tmp2);

                        } else if (op == 4 || (op == 5 && u)) {

                            /* Insert */

                            switch (size) {

                            case 0:

                                if (op == 4)

                                    mask = 0xff >> -shift;

                                else

                                    mask = (uint8_t)(0xff << shift);

                                mask |= mask << 8;

                                mask |= mask << 16;

                                break;

                            case 1:

                                if (op == 4)

                                    mask = 0xffff >> -shift;

                                else

                                    mask = (uint16_t)(0xffff << shift);

                                mask |= mask << 16;

                                break;

                            case 2:

                                if (shift < -31 || shift > 31) {

                                    mask = 0;

                                } else {

                                    if (op == 4)

                                        mask = 0xffffffffu >> -shift;

                                    else

                                        mask = 0xffffffffu << shift;

                                }

                                break;

                            default:

                                abort();

                            }

                            tmp2 = neon_load_reg(rd, pass);

                            tcg_gen_andi_i32(tmp, tmp, mask);

                            tcg_gen_andi_i32(tmp2, tmp2, ~mask);

                            tcg_gen_or_i32(tmp, tmp, tmp2);

                            tcg_temp_free_i32(tmp2);

                        }

                        neon_store_reg(rd, pass, tmp);

                    }

                } /* for pass */

            } else if (op < 10) {

                /* Shift by immediate and narrow:

                   VSHRN, VRSHRN, VQSHRN, VQRSHRN.  */

                int input_unsigned = (op == 8) ? !u : u;

                if (rm & 1) {

                    return 1;

                }

                shift = shift - (1 << (size + 3));

                size++;

                if (size == 3) {

                    tmp64 = tcg_const_i64(shift);

                    neon_load_reg64(cpu_V0, rm);

                    neon_load_reg64(cpu_V1, rm + 1);

                    for (pass = 0; pass < 2; pass++) {

                        TCGv_i64 in;

                        if (pass == 0) {

                            in = cpu_V0;

                        } else {

                            in = cpu_V1;

                        }

                        if (q) {

                            if (input_unsigned) {

                                gen_helper_neon_rshl_u64(cpu_V0, in, tmp64);

                            } else {

                                gen_helper_neon_rshl_s64(cpu_V0, in, tmp64);

                            }

                        } else {

                            if (input_unsigned) {

                                gen_helper_neon_shl_u64(cpu_V0, in, tmp64);

                            } else {

                                gen_helper_neon_shl_s64(cpu_V0, in, tmp64);

                            }

                        }

                        tmp = tcg_temp_new_i32();

                        gen_neon_narrow_op(op == 8, u, size - 1, tmp, cpu_V0);

                        neon_store_reg(rd, pass, tmp);

                    } /* for pass */

                    tcg_temp_free_i64(tmp64);

                } else {

                    if (size == 1) {

                        imm = (uint16_t)shift;

                        imm |= imm << 16;

                    } else {

                        /* size == 2 */

                        imm = (uint32_t)shift;

                    }

                    tmp2 = tcg_const_i32(imm);

                    tmp4 = neon_load_reg(rm + 1, 0);

                    tmp5 = neon_load_reg(rm + 1, 1);

                    for (pass = 0; pass < 2; pass++) {

                        if (pass == 0) {

                            tmp = neon_load_reg(rm, 0);

                        } else {

                            tmp = tmp4;

                        }

                        gen_neon_shift_narrow(size, tmp, tmp2, q,

                                              input_unsigned);

                        if (pass == 0) {

                            tmp3 = neon_load_reg(rm, 1);

                        } else {

                            tmp3 = tmp5;

                        }

                        gen_neon_shift_narrow(size, tmp3, tmp2, q,

                                              input_unsigned);

                        tcg_gen_concat_i32_i64(cpu_V0, tmp, tmp3);

                        tcg_temp_free_i32(tmp);

                        tcg_temp_free_i32(tmp3);

                        tmp = tcg_temp_new_i32();

                        gen_neon_narrow_op(op == 8, u, size - 1, tmp, cpu_V0);

                        neon_store_reg(rd, pass, tmp);

                    } /* for pass */

                    tcg_temp_free_i32(tmp2);

                }

            } else if (op == 10) {

                /* VSHLL, VMOVL */

                if (q || (rd & 1)) {

                    return 1;

                }

                tmp = neon_load_reg(rm, 0);

                tmp2 = neon_load_reg(rm, 1);

                for (pass = 0; pass < 2; pass++) {

                    if (pass == 1)

                        tmp = tmp2;



                    gen_neon_widen(cpu_V0, tmp, size, u);



                    if (shift != 0) {

                        /* The shift is less than the width of the source

                           type, so we can just shift the whole register.  */

                        tcg_gen_shli_i64(cpu_V0, cpu_V0, shift);

                        /* Widen the result of shift: we need to clear

                         * the potential overflow bits resulting from

                         * left bits of the narrow input appearing as

                         * right bits of left the neighbour narrow

                         * input.  */

                        if (size < 2 || !u) {

                            uint64_t imm64;

                            if (size == 0) {

                                imm = (0xffu >> (8 - shift));

                                imm |= imm << 16;

                            } else if (size == 1) {

                                imm = 0xffff >> (16 - shift);

                            } else {

                                /* size == 2 */

                                imm = 0xffffffff >> (32 - shift);

                            }

                            if (size < 2) {

                                imm64 = imm | (((uint64_t)imm) << 32);

                            } else {

                                imm64 = imm;

                            }

                            tcg_gen_andi_i64(cpu_V0, cpu_V0, ~imm64);

                        }

                    }

                    neon_store_reg64(cpu_V0, rd + pass);

                }

            } else if (op >= 14) {

                /* VCVT fixed-point.  */

                if (!(insn & (1 << 21)) || (q && ((rd | rm) & 1))) {

                    return 1;

                }

                /* We have already masked out the must-be-1 top bit of imm6,

                 * hence this 32-shift where the ARM ARM has 64-imm6.

                 */

                shift = 32 - shift;

                for (pass = 0; pass < (q ? 4 : 2); pass++) {

                    tcg_gen_ld_f32(cpu_F0s, cpu_env, neon_reg_offset(rm, pass));

                    if (!(op & 1)) {

                        if (u)

                            gen_vfp_ulto(0, shift);

                        else

                            gen_vfp_slto(0, shift);

                    } else {

                        if (u)

                            gen_vfp_toul(0, shift);

                        else

                            gen_vfp_tosl(0, shift);

                    }

                    tcg_gen_st_f32(cpu_F0s, cpu_env, neon_reg_offset(rd, pass));

                }

            } else {

                return 1;

            }

        } else { /* (insn & 0x00380080) == 0 */

            int invert;

            if (q && (rd & 1)) {

                return 1;

            }



            op = (insn >> 8) & 0xf;

            /* One register and immediate.  */

            imm = (u << 7) | ((insn >> 12) & 0x70) | (insn & 0xf);

            invert = (insn & (1 << 5)) != 0;

            /* Note that op = 2,3,4,5,6,7,10,11,12,13 imm=0 is UNPREDICTABLE.

             * We choose to not special-case this and will behave as if a

             * valid constant encoding of 0 had been given.

             */

            switch (op) {

            case 0: case 1:

                /* no-op */

                break;

            case 2: case 3:

                imm <<= 8;

                break;

            case 4: case 5:

                imm <<= 16;

                break;

            case 6: case 7:

                imm <<= 24;

                break;

            case 8: case 9:

                imm |= imm << 16;

                break;

            case 10: case 11:

                imm = (imm << 8) | (imm << 24);

                break;

            case 12:

                imm = (imm << 8) | 0xff;

                break;

            case 13:

                imm = (imm << 16) | 0xffff;

                break;

            case 14:

                imm |= (imm << 8) | (imm << 16) | (imm << 24);

                if (invert)

                    imm = ~imm;

                break;

            case 15:

                if (invert) {

                    return 1;

                }

                imm = ((imm & 0x80) << 24) | ((imm & 0x3f) << 19)

                      | ((imm & 0x40) ? (0x1f << 25) : (1 << 30));

                break;

            }

            if (invert)

                imm = ~imm;



            for (pass = 0; pass < (q ? 4 : 2); pass++) {

                if (op & 1 && op < 12) {

                    tmp = neon_load_reg(rd, pass);

                    if (invert) {

                        /* The immediate value has already been inverted, so

                           BIC becomes AND.  */

                        tcg_gen_andi_i32(tmp, tmp, imm);

                    } else {

                        tcg_gen_ori_i32(tmp, tmp, imm);

                    }

                } else {

                    /* VMOV, VMVN.  */

                    tmp = tcg_temp_new_i32();

                    if (op == 14 && invert) {

                        int n;

                        uint32_t val;

                        val = 0;

                        for (n = 0; n < 4; n++) {

                            if (imm & (1 << (n + (pass & 1) * 4)))

                                val |= 0xff << (n * 8);

                        }

                        tcg_gen_movi_i32(tmp, val);

                    } else {

                        tcg_gen_movi_i32(tmp, imm);

                    }

                }

                neon_store_reg(rd, pass, tmp);

            }

        }

    } else { /* (insn & 0x00800010 == 0x00800000) */

        if (size != 3) {

            op = (insn >> 8) & 0xf;

            if ((insn & (1 << 6)) == 0) {

                /* Three registers of different lengths.  */

                int src1_wide;

                int src2_wide;

                int prewiden;

                /* undefreq: bit 0 : UNDEF if size != 0

                 *           bit 1 : UNDEF if size == 0

                 *           bit 2 : UNDEF if U == 1

                 * Note that [1:0] set implies 'always UNDEF'

                 */

                int undefreq;

                /* prewiden, src1_wide, src2_wide, undefreq */

                static const int neon_3reg_wide[16][4] = {

                    {1, 0, 0, 0}, /* VADDL */

                    {1, 1, 0, 0}, /* VADDW */

                    {1, 0, 0, 0}, /* VSUBL */

                    {1, 1, 0, 0}, /* VSUBW */

                    {0, 1, 1, 0}, /* VADDHN */

                    {0, 0, 0, 0}, /* VABAL */

                    {0, 1, 1, 0}, /* VSUBHN */

                    {0, 0, 0, 0}, /* VABDL */

                    {0, 0, 0, 0}, /* VMLAL */

                    {0, 0, 0, 6}, /* VQDMLAL */

                    {0, 0, 0, 0}, /* VMLSL */

                    {0, 0, 0, 6}, /* VQDMLSL */

                    {0, 0, 0, 0}, /* Integer VMULL */

                    {0, 0, 0, 2}, /* VQDMULL */

                    {0, 0, 0, 5}, /* Polynomial VMULL */

                    {0, 0, 0, 3}, /* Reserved: always UNDEF */

                };



                prewiden = neon_3reg_wide[op][0];

                src1_wide = neon_3reg_wide[op][1];

                src2_wide = neon_3reg_wide[op][2];

                undefreq = neon_3reg_wide[op][3];



                if (((undefreq & 1) && (size != 0)) ||

                    ((undefreq & 2) && (size == 0)) ||

                    ((undefreq & 4) && u)) {

                    return 1;

                }

                if ((src1_wide && (rn & 1)) ||

                    (src2_wide && (rm & 1)) ||

                    (!src2_wide && (rd & 1))) {

                    return 1;

                }



                /* Avoid overlapping operands.  Wide source operands are

                   always aligned so will never overlap with wide

                   destinations in problematic ways.  */

                if (rd == rm && !src2_wide) {

                    tmp = neon_load_reg(rm, 1);

                    neon_store_scratch(2, tmp);

                } else if (rd == rn && !src1_wide) {

                    tmp = neon_load_reg(rn, 1);

                    neon_store_scratch(2, tmp);

                }

                TCGV_UNUSED(tmp3);

                for (pass = 0; pass < 2; pass++) {

                    if (src1_wide) {

                        neon_load_reg64(cpu_V0, rn + pass);

                        TCGV_UNUSED(tmp);

                    } else {

                        if (pass == 1 && rd == rn) {

                            tmp = neon_load_scratch(2);

                        } else {

                            tmp = neon_load_reg(rn, pass);

                        }

                        if (prewiden) {

                            gen_neon_widen(cpu_V0, tmp, size, u);

                        }

                    }

                    if (src2_wide) {

                        neon_load_reg64(cpu_V1, rm + pass);

                        TCGV_UNUSED(tmp2);

                    } else {

                        if (pass == 1 && rd == rm) {

                            tmp2 = neon_load_scratch(2);

                        } else {

                            tmp2 = neon_load_reg(rm, pass);

                        }

                        if (prewiden) {

                            gen_neon_widen(cpu_V1, tmp2, size, u);

                        }

                    }

                    switch (op) {

                    case 0: case 1: case 4: /* VADDL, VADDW, VADDHN, VRADDHN */

                        gen_neon_addl(size);

                        break;

                    case 2: case 3: case 6: /* VSUBL, VSUBW, VSUBHN, VRSUBHN */

                        gen_neon_subl(size);

                        break;

                    case 5: case 7: /* VABAL, VABDL */

                        switch ((size << 1) | u) {

                        case 0:

                            gen_helper_neon_abdl_s16(cpu_V0, tmp, tmp2);

                            break;

                        case 1:

                            gen_helper_neon_abdl_u16(cpu_V0, tmp, tmp2);

                            break;

                        case 2:

                            gen_helper_neon_abdl_s32(cpu_V0, tmp, tmp2);

                            break;

                        case 3:

                            gen_helper_neon_abdl_u32(cpu_V0, tmp, tmp2);

                            break;

                        case 4:

                            gen_helper_neon_abdl_s64(cpu_V0, tmp, tmp2);

                            break;

                        case 5:

                            gen_helper_neon_abdl_u64(cpu_V0, tmp, tmp2);

                            break;

                        default: abort();

                        }

                        tcg_temp_free_i32(tmp2);

                        tcg_temp_free_i32(tmp);

                        break;

                    case 8: case 9: case 10: case 11: case 12: case 13:

                        /* VMLAL, VQDMLAL, VMLSL, VQDMLSL, VMULL, VQDMULL */

                        gen_neon_mull(cpu_V0, tmp, tmp2, size, u);

                        break;

                    case 14: /* Polynomial VMULL */

                        gen_helper_neon_mull_p8(cpu_V0, tmp, tmp2);

                        tcg_temp_free_i32(tmp2);

                        tcg_temp_free_i32(tmp);

                        break;

                    default: /* 15 is RESERVED: caught earlier  */

                        abort();

                    }

                    if (op == 13) {

                        /* VQDMULL */

                        gen_neon_addl_saturate(cpu_V0, cpu_V0, size);

                        neon_store_reg64(cpu_V0, rd + pass);

                    } else if (op == 5 || (op >= 8 && op <= 11)) {

                        /* Accumulate.  */

                        neon_load_reg64(cpu_V1, rd + pass);

                        switch (op) {

                        case 10: /* VMLSL */

                            gen_neon_negl(cpu_V0, size);

                            /* Fall through */

                        case 5: case 8: /* VABAL, VMLAL */

                            gen_neon_addl(size);

                            break;

                        case 9: case 11: /* VQDMLAL, VQDMLSL */

                            gen_neon_addl_saturate(cpu_V0, cpu_V0, size);

                            if (op == 11) {

                                gen_neon_negl(cpu_V0, size);

                            }

                            gen_neon_addl_saturate(cpu_V0, cpu_V1, size);

                            break;

                        default:

                            abort();

                        }

                        neon_store_reg64(cpu_V0, rd + pass);

                    } else if (op == 4 || op == 6) {

                        /* Narrowing operation.  */

                        tmp = tcg_temp_new_i32();

                        if (!u) {

                            switch (size) {

                            case 0:

                                gen_helper_neon_narrow_high_u8(tmp, cpu_V0);

                                break;

                            case 1:

                                gen_helper_neon_narrow_high_u16(tmp, cpu_V0);

                                break;

                            case 2:

                                tcg_gen_shri_i64(cpu_V0, cpu_V0, 32);

                                tcg_gen_trunc_i64_i32(tmp, cpu_V0);

                                break;

                            default: abort();

                            }

                        } else {

                            switch (size) {

                            case 0:

                                gen_helper_neon_narrow_round_high_u8(tmp, cpu_V0);

                                break;

                            case 1:

                                gen_helper_neon_narrow_round_high_u16(tmp, cpu_V0);

                                break;

                            case 2:

                                tcg_gen_addi_i64(cpu_V0, cpu_V0, 1u << 31);

                                tcg_gen_shri_i64(cpu_V0, cpu_V0, 32);

                                tcg_gen_trunc_i64_i32(tmp, cpu_V0);

                                break;

                            default: abort();

                            }

                        }

                        if (pass == 0) {

                            tmp3 = tmp;

                        } else {

                            neon_store_reg(rd, 0, tmp3);

                            neon_store_reg(rd, 1, tmp);

                        }

                    } else {

                        /* Write back the result.  */

                        neon_store_reg64(cpu_V0, rd + pass);

                    }

                }

            } else {

                /* Two registers and a scalar. NB that for ops of this form

                 * the ARM ARM labels bit 24 as Q, but it is in our variable

                 * 'u', not 'q'.

                 */

                if (size == 0) {

                    return 1;

                }

                switch (op) {

                case 1: /* Float VMLA scalar */

                case 5: /* Floating point VMLS scalar */

                case 9: /* Floating point VMUL scalar */

                    if (size == 1) {

                        return 1;

                    }

                    /* fall through */

                case 0: /* Integer VMLA scalar */

                case 4: /* Integer VMLS scalar */

                case 8: /* Integer VMUL scalar */

                case 12: /* VQDMULH scalar */

                case 13: /* VQRDMULH scalar */

                    if (u && ((rd | rn) & 1)) {

                        return 1;

                    }

                    tmp = neon_get_scalar(size, rm);

                    neon_store_scratch(0, tmp);

                    for (pass = 0; pass < (u ? 4 : 2); pass++) {

                        tmp = neon_load_scratch(0);

                        tmp2 = neon_load_reg(rn, pass);

                        if (op == 12) {

                            if (size == 1) {

                                gen_helper_neon_qdmulh_s16(tmp, tmp, tmp2);

                            } else {

                                gen_helper_neon_qdmulh_s32(tmp, tmp, tmp2);

                            }

                        } else if (op == 13) {

                            if (size == 1) {

                                gen_helper_neon_qrdmulh_s16(tmp, tmp, tmp2);

                            } else {

                                gen_helper_neon_qrdmulh_s32(tmp, tmp, tmp2);

                            }

                        } else if (op & 1) {

                            gen_helper_neon_mul_f32(tmp, tmp, tmp2);

                        } else {

                            switch (size) {

                            case 0: gen_helper_neon_mul_u8(tmp, tmp, tmp2); break;

                            case 1: gen_helper_neon_mul_u16(tmp, tmp, tmp2); break;

                            case 2: tcg_gen_mul_i32(tmp, tmp, tmp2); break;

                            default: abort();

                            }

                        }

                        tcg_temp_free_i32(tmp2);

                        if (op < 8) {

                            /* Accumulate.  */

                            tmp2 = neon_load_reg(rd, pass);

                            switch (op) {

                            case 0:

                                gen_neon_add(size, tmp, tmp2);

                                break;

                            case 1:

                                gen_helper_neon_add_f32(tmp, tmp, tmp2);

                                break;

                            case 4:

                                gen_neon_rsb(size, tmp, tmp2);

                                break;

                            case 5:

                                gen_helper_neon_sub_f32(tmp, tmp2, tmp);

                                break;

                            default:

                                abort();

                            }

                            tcg_temp_free_i32(tmp2);

                        }

                        neon_store_reg(rd, pass, tmp);

                    }

                    break;

                case 3: /* VQDMLAL scalar */

                case 7: /* VQDMLSL scalar */

                case 11: /* VQDMULL scalar */

                    if (u == 1) {

                        return 1;

                    }

                    /* fall through */

                case 2: /* VMLAL sclar */

                case 6: /* VMLSL scalar */

                case 10: /* VMULL scalar */

                    if (rd & 1) {

                        return 1;

                    }

                    tmp2 = neon_get_scalar(size, rm);

                    /* We need a copy of tmp2 because gen_neon_mull

                     * deletes it during pass 0.  */

                    tmp4 = tcg_temp_new_i32();

                    tcg_gen_mov_i32(tmp4, tmp2);

                    tmp3 = neon_load_reg(rn, 1);



                    for (pass = 0; pass < 2; pass++) {

                        if (pass == 0) {

                            tmp = neon_load_reg(rn, 0);

                        } else {

                            tmp = tmp3;

                            tmp2 = tmp4;

                        }

                        gen_neon_mull(cpu_V0, tmp, tmp2, size, u);

                        if (op != 11) {

                            neon_load_reg64(cpu_V1, rd + pass);

                        }

                        switch (op) {

                        case 6:

                            gen_neon_negl(cpu_V0, size);

                            /* Fall through */

                        case 2:

                            gen_neon_addl(size);

                            break;

                        case 3: case 7:

                            gen_neon_addl_saturate(cpu_V0, cpu_V0, size);

                            if (op == 7) {

                                gen_neon_negl(cpu_V0, size);

                            }

                            gen_neon_addl_saturate(cpu_V0, cpu_V1, size);

                            break;

                        case 10:

                            /* no-op */

                            break;

                        case 11:

                            gen_neon_addl_saturate(cpu_V0, cpu_V0, size);

                            break;

                        default:

                            abort();

                        }

                        neon_store_reg64(cpu_V0, rd + pass);

                    }





                    break;

                default: /* 14 and 15 are RESERVED */

                    return 1;

                }

            }

        } else { /* size == 3 */

            if (!u) {

                /* Extract.  */

                imm = (insn >> 8) & 0xf;



                if (imm > 7 && !q)

                    return 1;



                if (q && ((rd | rn | rm) & 1)) {

                    return 1;

                }



                if (imm == 0) {

                    neon_load_reg64(cpu_V0, rn);

                    if (q) {

                        neon_load_reg64(cpu_V1, rn + 1);

                    }

                } else if (imm == 8) {

                    neon_load_reg64(cpu_V0, rn + 1);

                    if (q) {

                        neon_load_reg64(cpu_V1, rm);

                    }

                } else if (q) {

                    tmp64 = tcg_temp_new_i64();

                    if (imm < 8) {

                        neon_load_reg64(cpu_V0, rn);

                        neon_load_reg64(tmp64, rn + 1);

                    } else {

                        neon_load_reg64(cpu_V0, rn + 1);

                        neon_load_reg64(tmp64, rm);

                    }

                    tcg_gen_shri_i64(cpu_V0, cpu_V0, (imm & 7) * 8);

                    tcg_gen_shli_i64(cpu_V1, tmp64, 64 - ((imm & 7) * 8));

                    tcg_gen_or_i64(cpu_V0, cpu_V0, cpu_V1);

                    if (imm < 8) {

                        neon_load_reg64(cpu_V1, rm);

                    } else {

                        neon_load_reg64(cpu_V1, rm + 1);

                        imm -= 8;

                    }

                    tcg_gen_shli_i64(cpu_V1, cpu_V1, 64 - (imm * 8));

                    tcg_gen_shri_i64(tmp64, tmp64, imm * 8);

                    tcg_gen_or_i64(cpu_V1, cpu_V1, tmp64);

                    tcg_temp_free_i64(tmp64);

                } else {

                    /* BUGFIX */

                    neon_load_reg64(cpu_V0, rn);

                    tcg_gen_shri_i64(cpu_V0, cpu_V0, imm * 8);

                    neon_load_reg64(cpu_V1, rm);

                    tcg_gen_shli_i64(cpu_V1, cpu_V1, 64 - (imm * 8));

                    tcg_gen_or_i64(cpu_V0, cpu_V0, cpu_V1);

                }

                neon_store_reg64(cpu_V0, rd);

                if (q) {

                    neon_store_reg64(cpu_V1, rd + 1);

                }

            } else if ((insn & (1 << 11)) == 0) {

                /* Two register misc.  */

                op = ((insn >> 12) & 0x30) | ((insn >> 7) & 0xf);

                size = (insn >> 18) & 3;

                /* UNDEF for unknown op values and bad op-size combinations */

                if ((neon_2rm_sizes[op] & (1 << size)) == 0) {

                    return 1;

                }

                switch (op) {

                case NEON_2RM_VREV64:

                    for (pass = 0; pass < (q ? 2 : 1); pass++) {

                        tmp = neon_load_reg(rm, pass * 2);

                        tmp2 = neon_load_reg(rm, pass * 2 + 1);

                        switch (size) {

                        case 0: tcg_gen_bswap32_i32(tmp, tmp); break;

                        case 1: gen_swap_half(tmp); break;

                        case 2: /* no-op */ break;

                        default: abort();

                        }

                        neon_store_reg(rd, pass * 2 + 1, tmp);

                        if (size == 2) {

                            neon_store_reg(rd, pass * 2, tmp2);

                        } else {

                            switch (size) {

                            case 0: tcg_gen_bswap32_i32(tmp2, tmp2); break;

                            case 1: gen_swap_half(tmp2); break;

                            default: abort();

                            }

                            neon_store_reg(rd, pass * 2, tmp2);

                        }

                    }

                    break;

                case NEON_2RM_VPADDL: case NEON_2RM_VPADDL_U:

                case NEON_2RM_VPADAL: case NEON_2RM_VPADAL_U:

                    for (pass = 0; pass < q + 1; pass++) {

                        tmp = neon_load_reg(rm, pass * 2);

                        gen_neon_widen(cpu_V0, tmp, size, op & 1);

                        tmp = neon_load_reg(rm, pass * 2 + 1);

                        gen_neon_widen(cpu_V1, tmp, size, op & 1);

                        switch (size) {

                        case 0: gen_helper_neon_paddl_u16(CPU_V001); break;

                        case 1: gen_helper_neon_paddl_u32(CPU_V001); break;

                        case 2: tcg_gen_add_i64(CPU_V001); break;

                        default: abort();

                        }

                        if (op >= NEON_2RM_VPADAL) {

                            /* Accumulate.  */

                            neon_load_reg64(cpu_V1, rd + pass);

                            gen_neon_addl(size);

                        }

                        neon_store_reg64(cpu_V0, rd + pass);

                    }

                    break;

                case NEON_2RM_VTRN:

                    if (size == 2) {

                        int n;

                        for (n = 0; n < (q ? 4 : 2); n += 2) {

                            tmp = neon_load_reg(rm, n);

                            tmp2 = neon_load_reg(rd, n + 1);

                            neon_store_reg(rm, n, tmp2);

                            neon_store_reg(rd, n + 1, tmp);

                        }

                    } else {

                        goto elementwise;

                    }

                    break;

                case NEON_2RM_VUZP:

                    if (gen_neon_unzip(rd, rm, size, q)) {

                        return 1;

                    }

                    break;

                case NEON_2RM_VZIP:

                    if (gen_neon_zip(rd, rm, size, q)) {

                        return 1;

                    }

                    break;

                case NEON_2RM_VMOVN: case NEON_2RM_VQMOVN:

                    /* also VQMOVUN; op field and mnemonics don't line up */

                    TCGV_UNUSED(tmp2);

                    for (pass = 0; pass < 2; pass++) {

                        neon_load_reg64(cpu_V0, rm + pass);

                        tmp = tcg_temp_new_i32();

                        gen_neon_narrow_op(op == NEON_2RM_VMOVN, q, size,

                                           tmp, cpu_V0);

                        if (pass == 0) {

                            tmp2 = tmp;

                        } else {

                            neon_store_reg(rd, 0, tmp2);

                            neon_store_reg(rd, 1, tmp);

                        }

                    }

                    break;

                case NEON_2RM_VSHLL:

                    if (q) {

                        return 1;

                    }

                    tmp = neon_load_reg(rm, 0);

                    tmp2 = neon_load_reg(rm, 1);

                    for (pass = 0; pass < 2; pass++) {

                        if (pass == 1)

                            tmp = tmp2;

                        gen_neon_widen(cpu_V0, tmp, size, 1);

                        tcg_gen_shli_i64(cpu_V0, cpu_V0, 8 << size);

                        neon_store_reg64(cpu_V0, rd + pass);

                    }

                    break;

                case NEON_2RM_VCVT_F16_F32:

                    if (!arm_feature(env, ARM_FEATURE_VFP_FP16))

                      return 1;

                    tmp = tcg_temp_new_i32();

                    tmp2 = tcg_temp_new_i32();

                    tcg_gen_ld_f32(cpu_F0s, cpu_env, neon_reg_offset(rm, 0));

                    gen_helper_neon_fcvt_f32_to_f16(tmp, cpu_F0s, cpu_env);

                    tcg_gen_ld_f32(cpu_F0s, cpu_env, neon_reg_offset(rm, 1));

                    gen_helper_neon_fcvt_f32_to_f16(tmp2, cpu_F0s, cpu_env);

                    tcg_gen_shli_i32(tmp2, tmp2, 16);

                    tcg_gen_or_i32(tmp2, tmp2, tmp);

                    tcg_gen_ld_f32(cpu_F0s, cpu_env, neon_reg_offset(rm, 2));

                    gen_helper_neon_fcvt_f32_to_f16(tmp, cpu_F0s, cpu_env);

                    tcg_gen_ld_f32(cpu_F0s, cpu_env, neon_reg_offset(rm, 3));

                    neon_store_reg(rd, 0, tmp2);

                    tmp2 = tcg_temp_new_i32();

                    gen_helper_neon_fcvt_f32_to_f16(tmp2, cpu_F0s, cpu_env);

                    tcg_gen_shli_i32(tmp2, tmp2, 16);

                    tcg_gen_or_i32(tmp2, tmp2, tmp);

                    neon_store_reg(rd, 1, tmp2);

                    tcg_temp_free_i32(tmp);

                    break;

                case NEON_2RM_VCVT_F32_F16:

                    if (!arm_feature(env, ARM_FEATURE_VFP_FP16))

                      return 1;

                    tmp3 = tcg_temp_new_i32();

                    tmp = neon_load_reg(rm, 0);

                    tmp2 = neon_load_reg(rm, 1);

                    tcg_gen_ext16u_i32(tmp3, tmp);

                    gen_helper_neon_fcvt_f16_to_f32(cpu_F0s, tmp3, cpu_env);

                    tcg_gen_st_f32(cpu_F0s, cpu_env, neon_reg_offset(rd, 0));

                    tcg_gen_shri_i32(tmp3, tmp, 16);

                    gen_helper_neon_fcvt_f16_to_f32(cpu_F0s, tmp3, cpu_env);

                    tcg_gen_st_f32(cpu_F0s, cpu_env, neon_reg_offset(rd, 1));

                    tcg_temp_free_i32(tmp);

                    tcg_gen_ext16u_i32(tmp3, tmp2);

                    gen_helper_neon_fcvt_f16_to_f32(cpu_F0s, tmp3, cpu_env);

                    tcg_gen_st_f32(cpu_F0s, cpu_env, neon_reg_offset(rd, 2));

                    tcg_gen_shri_i32(tmp3, tmp2, 16);

                    gen_helper_neon_fcvt_f16_to_f32(cpu_F0s, tmp3, cpu_env);

                    tcg_gen_st_f32(cpu_F0s, cpu_env, neon_reg_offset(rd, 3));

                    tcg_temp_free_i32(tmp2);

                    tcg_temp_free_i32(tmp3);

                    break;

                default:

                elementwise:

                    for (pass = 0; pass < (q ? 4 : 2); pass++) {

                        if (neon_2rm_is_float_op(op)) {

                            tcg_gen_ld_f32(cpu_F0s, cpu_env,

                                           neon_reg_offset(rm, pass));

                            TCGV_UNUSED(tmp);

                        } else {

                            tmp = neon_load_reg(rm, pass);

                        }

                        switch (op) {

                        case NEON_2RM_VREV32:

                            switch (size) {

                            case 0: tcg_gen_bswap32_i32(tmp, tmp); break;

                            case 1: gen_swap_half(tmp); break;

                            default: abort();

                            }

                            break;

                        case NEON_2RM_VREV16:

                            gen_rev16(tmp);

                            break;

                        case NEON_2RM_VCLS:

                            switch (size) {

                            case 0: gen_helper_neon_cls_s8(tmp, tmp); break;

                            case 1: gen_helper_neon_cls_s16(tmp, tmp); break;

                            case 2: gen_helper_neon_cls_s32(tmp, tmp); break;

                            default: abort();

                            }

                            break;

                        case NEON_2RM_VCLZ:

                            switch (size) {

                            case 0: gen_helper_neon_clz_u8(tmp, tmp); break;

                            case 1: gen_helper_neon_clz_u16(tmp, tmp); break;

                            case 2: gen_helper_clz(tmp, tmp); break;

                            default: abort();

                            }

                            break;

                        case NEON_2RM_VCNT:

                            gen_helper_neon_cnt_u8(tmp, tmp);

                            break;

                        case NEON_2RM_VMVN:

                            tcg_gen_not_i32(tmp, tmp);

                            break;

                        case NEON_2RM_VQABS:

                            switch (size) {

                            case 0: gen_helper_neon_qabs_s8(tmp, tmp); break;

                            case 1: gen_helper_neon_qabs_s16(tmp, tmp); break;

                            case 2: gen_helper_neon_qabs_s32(tmp, tmp); break;

                            default: abort();

                            }

                            break;

                        case NEON_2RM_VQNEG:

                            switch (size) {

                            case 0: gen_helper_neon_qneg_s8(tmp, tmp); break;

                            case 1: gen_helper_neon_qneg_s16(tmp, tmp); break;

                            case 2: gen_helper_neon_qneg_s32(tmp, tmp); break;

                            default: abort();

                            }

                            break;

                        case NEON_2RM_VCGT0: case NEON_2RM_VCLE0:

                            tmp2 = tcg_const_i32(0);

                            switch(size) {

                            case 0: gen_helper_neon_cgt_s8(tmp, tmp, tmp2); break;

                            case 1: gen_helper_neon_cgt_s16(tmp, tmp, tmp2); break;

                            case 2: gen_helper_neon_cgt_s32(tmp, tmp, tmp2); break;

                            default: abort();

                            }

                            tcg_temp_free(tmp2);

                            if (op == NEON_2RM_VCLE0) {

                                tcg_gen_not_i32(tmp, tmp);

                            }

                            break;

                        case NEON_2RM_VCGE0: case NEON_2RM_VCLT0:

                            tmp2 = tcg_const_i32(0);

                            switch(size) {

                            case 0: gen_helper_neon_cge_s8(tmp, tmp, tmp2); break;

                            case 1: gen_helper_neon_cge_s16(tmp, tmp, tmp2); break;

                            case 2: gen_helper_neon_cge_s32(tmp, tmp, tmp2); break;

                            default: abort();

                            }

                            tcg_temp_free(tmp2);

                            if (op == NEON_2RM_VCLT0) {

                                tcg_gen_not_i32(tmp, tmp);

                            }

                            break;

                        case NEON_2RM_VCEQ0:

                            tmp2 = tcg_const_i32(0);

                            switch(size) {

                            case 0: gen_helper_neon_ceq_u8(tmp, tmp, tmp2); break;

                            case 1: gen_helper_neon_ceq_u16(tmp, tmp, tmp2); break;

                            case 2: gen_helper_neon_ceq_u32(tmp, tmp, tmp2); break;

                            default: abort();

                            }

                            tcg_temp_free(tmp2);

                            break;

                        case NEON_2RM_VABS:

                            switch(size) {

                            case 0: gen_helper_neon_abs_s8(tmp, tmp); break;

                            case 1: gen_helper_neon_abs_s16(tmp, tmp); break;

                            case 2: tcg_gen_abs_i32(tmp, tmp); break;

                            default: abort();

                            }

                            break;

                        case NEON_2RM_VNEG:

                            tmp2 = tcg_const_i32(0);

                            gen_neon_rsb(size, tmp, tmp2);

                            tcg_temp_free(tmp2);

                            break;

                        case NEON_2RM_VCGT0_F:

                            tmp2 = tcg_const_i32(0);

                            gen_helper_neon_cgt_f32(tmp, tmp, tmp2);

                            tcg_temp_free(tmp2);

                            break;

                        case NEON_2RM_VCGE0_F:

                            tmp2 = tcg_const_i32(0);

                            gen_helper_neon_cge_f32(tmp, tmp, tmp2);

                            tcg_temp_free(tmp2);

                            break;

                        case NEON_2RM_VCEQ0_F:

                            tmp2 = tcg_const_i32(0);

                            gen_helper_neon_ceq_f32(tmp, tmp, tmp2);

                            tcg_temp_free(tmp2);

                            break;

                        case NEON_2RM_VCLE0_F:

                            tmp2 = tcg_const_i32(0);

                            gen_helper_neon_cge_f32(tmp, tmp2, tmp);

                            tcg_temp_free(tmp2);

                            break;

                        case NEON_2RM_VCLT0_F:

                            tmp2 = tcg_const_i32(0);

                            gen_helper_neon_cgt_f32(tmp, tmp2, tmp);

                            tcg_temp_free(tmp2);

                            break;

                        case NEON_2RM_VABS_F:

                            gen_vfp_abs(0);

                            break;

                        case NEON_2RM_VNEG_F:

                            gen_vfp_neg(0);

                            break;

                        case NEON_2RM_VSWP:

                            tmp2 = neon_load_reg(rd, pass);

                            neon_store_reg(rm, pass, tmp2);

                            break;

                        case NEON_2RM_VTRN:

                            tmp2 = neon_load_reg(rd, pass);

                            switch (size) {

                            case 0: gen_neon_trn_u8(tmp, tmp2); break;

                            case 1: gen_neon_trn_u16(tmp, tmp2); break;

                            default: abort();

                            }

                            neon_store_reg(rm, pass, tmp2);

                            break;

                        case NEON_2RM_VRECPE:

                            gen_helper_recpe_u32(tmp, tmp, cpu_env);

                            break;

                        case NEON_2RM_VRSQRTE:

                            gen_helper_rsqrte_u32(tmp, tmp, cpu_env);

                            break;

                        case NEON_2RM_VRECPE_F:

                            gen_helper_recpe_f32(cpu_F0s, cpu_F0s, cpu_env);

                            break;

                        case NEON_2RM_VRSQRTE_F:

                            gen_helper_rsqrte_f32(cpu_F0s, cpu_F0s, cpu_env);

                            break;

                        case NEON_2RM_VCVT_FS: /* VCVT.F32.S32 */

                            gen_vfp_sito(0);

                            break;

                        case NEON_2RM_VCVT_FU: /* VCVT.F32.U32 */

                            gen_vfp_uito(0);

                            break;

                        case NEON_2RM_VCVT_SF: /* VCVT.S32.F32 */

                            gen_vfp_tosiz(0);

                            break;

                        case NEON_2RM_VCVT_UF: /* VCVT.U32.F32 */

                            gen_vfp_touiz(0);

                            break;

                        default:

                            /* Reserved op values were caught by the

                             * neon_2rm_sizes[] check earlier.

                             */

                            abort();

                        }

                        if (neon_2rm_is_float_op(op)) {

                            tcg_gen_st_f32(cpu_F0s, cpu_env,

                                           neon_reg_offset(rd, pass));

                        } else {

                            neon_store_reg(rd, pass, tmp);

                        }

                    }

                    break;

                }

            } else if ((insn & (1 << 10)) == 0) {

                /* VTBL, VTBX.  */

                int n = ((insn >> 5) & 0x18) + 8;

                if (insn & (1 << 6)) {

                    tmp = neon_load_reg(rd, 0);

                } else {

                    tmp = tcg_temp_new_i32();

                    tcg_gen_movi_i32(tmp, 0);

                }

                tmp2 = neon_load_reg(rm, 0);

                tmp4 = tcg_const_i32(rn);

                tmp5 = tcg_const_i32(n);

                gen_helper_neon_tbl(tmp2, tmp2, tmp, tmp4, tmp5);

                tcg_temp_free_i32(tmp);

                if (insn & (1 << 6)) {

                    tmp = neon_load_reg(rd, 1);

                } else {

                    tmp = tcg_temp_new_i32();

                    tcg_gen_movi_i32(tmp, 0);

                }

                tmp3 = neon_load_reg(rm, 1);

                gen_helper_neon_tbl(tmp3, tmp3, tmp, tmp4, tmp5);

                tcg_temp_free_i32(tmp5);

                tcg_temp_free_i32(tmp4);

                neon_store_reg(rd, 0, tmp2);

                neon_store_reg(rd, 1, tmp3);

                tcg_temp_free_i32(tmp);

            } else if ((insn & 0x380) == 0) {

                /* VDUP */

                if (insn & (1 << 19)) {

                    tmp = neon_load_reg(rm, 1);

                } else {

                    tmp = neon_load_reg(rm, 0);

                }

                if (insn & (1 << 16)) {

                    gen_neon_dup_u8(tmp, ((insn >> 17) & 3) * 8);

                } else if (insn & (1 << 17)) {

                    if ((insn >> 18) & 1)

                        gen_neon_dup_high16(tmp);

                    else

                        gen_neon_dup_low16(tmp);

                }

                for (pass = 0; pass < (q ? 4 : 2); pass++) {

                    tmp2 = tcg_temp_new_i32();

                    tcg_gen_mov_i32(tmp2, tmp);

                    neon_store_reg(rd, pass, tmp2);

                }

                tcg_temp_free_i32(tmp);

            } else {

                return 1;

            }

        }

    }

    return 0;

}
