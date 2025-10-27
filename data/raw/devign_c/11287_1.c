static always_inline int translate_one (DisasContext *ctx, uint32_t insn)

{

    uint32_t palcode;

    int32_t disp21, disp16, disp12;

    uint16_t fn11, fn16;

    uint8_t opc, ra, rb, rc, sbz, fpfn, fn7, fn2, islit;

    uint8_t lit;

    int ret;



    /* Decode all instruction fields */

    opc = insn >> 26;

    ra = (insn >> 21) & 0x1F;

    rb = (insn >> 16) & 0x1F;

    rc = insn & 0x1F;

    sbz = (insn >> 13) & 0x07;

    islit = (insn >> 12) & 1;

    if (rb == 31 && !islit) {

        islit = 1;

        lit = 0;

    } else

        lit = (insn >> 13) & 0xFF;

    palcode = insn & 0x03FFFFFF;

    disp21 = ((int32_t)((insn & 0x001FFFFF) << 11)) >> 11;

    disp16 = (int16_t)(insn & 0x0000FFFF);

    disp12 = (int32_t)((insn & 0x00000FFF) << 20) >> 20;

    fn16 = insn & 0x0000FFFF;

    fn11 = (insn >> 5) & 0x000007FF;

    fpfn = fn11 & 0x3F;

    fn7 = (insn >> 5) & 0x0000007F;

    fn2 = (insn >> 5) & 0x00000003;

    ret = 0;

#if defined ALPHA_DEBUG_DISAS

    if (logfile != NULL) {

        fprintf(logfile, "opc %02x ra %d rb %d rc %d disp16 %04x\n",

                opc, ra, rb, rc, disp16);

    }

#endif

    switch (opc) {

    case 0x00:

        /* CALL_PAL */

        if (palcode >= 0x80 && palcode < 0xC0) {

            /* Unprivileged PAL call */

            gen_excp(ctx, EXCP_CALL_PAL + ((palcode & 0x1F) << 6), 0);

#if !defined (CONFIG_USER_ONLY)

        } else if (palcode < 0x40) {

            /* Privileged PAL code */

            if (ctx->mem_idx & 1)

                goto invalid_opc;

            else

                gen_excp(ctx, EXCP_CALL_PALP + ((palcode & 0x1F) << 6), 0);

#endif

        } else {

            /* Invalid PAL call */

            goto invalid_opc;

        }

        ret = 3;

        break;

    case 0x01:

        /* OPC01 */

        goto invalid_opc;

    case 0x02:

        /* OPC02 */

        goto invalid_opc;

    case 0x03:

        /* OPC03 */

        goto invalid_opc;

    case 0x04:

        /* OPC04 */

        goto invalid_opc;

    case 0x05:

        /* OPC05 */

        goto invalid_opc;

    case 0x06:

        /* OPC06 */

        goto invalid_opc;

    case 0x07:

        /* OPC07 */

        goto invalid_opc;

    case 0x08:

        /* LDA */

        if (likely(ra != 31)) {

            if (rb != 31)

                tcg_gen_addi_i64(cpu_ir[ra], cpu_ir[rb], disp16);

            else

                tcg_gen_movi_i64(cpu_ir[ra], disp16);

        }

        break;

    case 0x09:

        /* LDAH */

        if (likely(ra != 31)) {

            if (rb != 31)

                tcg_gen_addi_i64(cpu_ir[ra], cpu_ir[rb], disp16 << 16);

            else

                tcg_gen_movi_i64(cpu_ir[ra], disp16 << 16);

        }

        break;

    case 0x0A:

        /* LDBU */

        if (!(ctx->amask & AMASK_BWX))

            goto invalid_opc;

        gen_load_mem(ctx, &tcg_gen_qemu_ld8u, ra, rb, disp16, 0, 0);

        break;

    case 0x0B:

        /* LDQ_U */

        gen_load_mem(ctx, &tcg_gen_qemu_ld64, ra, rb, disp16, 0, 1);

        break;

    case 0x0C:

        /* LDWU */

        if (!(ctx->amask & AMASK_BWX))

            goto invalid_opc;

        gen_load_mem(ctx, &tcg_gen_qemu_ld16u, ra, rb, disp16, 0, 1);

        break;

    case 0x0D:

        /* STW */

        gen_store_mem(ctx, &tcg_gen_qemu_st16, ra, rb, disp16, 0, 0);

        break;

    case 0x0E:

        /* STB */

        gen_store_mem(ctx, &tcg_gen_qemu_st8, ra, rb, disp16, 0, 0);

        break;

    case 0x0F:

        /* STQ_U */

        gen_store_mem(ctx, &tcg_gen_qemu_st64, ra, rb, disp16, 0, 1);

        break;

    case 0x10:

        switch (fn7) {

        case 0x00:

            /* ADDL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_addi_i64(cpu_ir[rc], cpu_ir[ra], lit);

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                    } else {

                        tcg_gen_add_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                    }

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    else

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x02:

            /* S4ADDL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 2);

                    if (islit)

                        tcg_gen_addi_i64(tmp, tmp, lit);

                    else

                        tcg_gen_add_i64(tmp, tmp, cpu_ir[rb]);

                    tcg_gen_ext32s_i64(cpu_ir[rc], tmp);

                    tcg_temp_free(tmp);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    else

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x09:

            /* SUBL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_subi_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    else

                        tcg_gen_sub_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    else {

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                }

            }

            break;

        case 0x0B:

            /* S4SUBL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 2);

                    if (islit)

                        tcg_gen_subi_i64(tmp, tmp, lit);

                    else

                        tcg_gen_sub_i64(tmp, tmp, cpu_ir[rb]);

                    tcg_gen_ext32s_i64(cpu_ir[rc], tmp);

                    tcg_temp_free(tmp);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    else {

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                    }

                }

            }

            break;

        case 0x0F:

            /* CMPBGE */

            gen_arith3(helper_cmpbge, ra, rb, rc, islit, lit);

            break;

        case 0x12:

            /* S8ADDL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 3);

                    if (islit)

                        tcg_gen_addi_i64(tmp, tmp, lit);

                    else

                        tcg_gen_add_i64(tmp, tmp, cpu_ir[rb]);

                    tcg_gen_ext32s_i64(cpu_ir[rc], tmp);

                    tcg_temp_free(tmp);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    else

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x1B:

            /* S8SUBL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 3);

                    if (islit)

                        tcg_gen_subi_i64(tmp, tmp, lit);

                    else

                       tcg_gen_sub_i64(tmp, tmp, cpu_ir[rb]);

                    tcg_gen_ext32s_i64(cpu_ir[rc], tmp);

                    tcg_temp_free(tmp);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    else

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                    }

                }

            }

            break;

        case 0x1D:

            /* CMPULT */

            gen_cmp(TCG_COND_LTU, ra, rb, rc, islit, lit);

            break;

        case 0x20:

            /* ADDQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_addi_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    else

                        tcg_gen_add_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    else

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x22:

            /* S4ADDQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 2);

                    if (islit)

                        tcg_gen_addi_i64(cpu_ir[rc], tmp, lit);

                    else

                        tcg_gen_add_i64(cpu_ir[rc], tmp, cpu_ir[rb]);

                    tcg_temp_free(tmp);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    else

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x29:

            /* SUBQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_subi_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    else

                        tcg_gen_sub_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    else

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x2B:

            /* S4SUBQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 2);

                    if (islit)

                        tcg_gen_subi_i64(cpu_ir[rc], tmp, lit);

                    else

                        tcg_gen_sub_i64(cpu_ir[rc], tmp, cpu_ir[rb]);

                    tcg_temp_free(tmp);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    else

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x2D:

            /* CMPEQ */

            gen_cmp(TCG_COND_EQ, ra, rb, rc, islit, lit);

            break;

        case 0x32:

            /* S8ADDQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 3);

                    if (islit)

                        tcg_gen_addi_i64(cpu_ir[rc], tmp, lit);

                    else

                        tcg_gen_add_i64(cpu_ir[rc], tmp, cpu_ir[rb]);

                    tcg_temp_free(tmp);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    else

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x3B:

            /* S8SUBQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 3);

                    if (islit)

                        tcg_gen_subi_i64(cpu_ir[rc], tmp, lit);

                    else

                        tcg_gen_sub_i64(cpu_ir[rc], tmp, cpu_ir[rb]);

                    tcg_temp_free(tmp);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    else

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x3D:

            /* CMPULE */

            gen_cmp(TCG_COND_LEU, ra, rb, rc, islit, lit);

            break;

        case 0x40:

            /* ADDL/V */

            gen_arith3(helper_addlv, ra, rb, rc, islit, lit);

            break;

        case 0x49:

            /* SUBL/V */

            gen_arith3(helper_sublv, ra, rb, rc, islit, lit);

            break;

        case 0x4D:

            /* CMPLT */

            gen_cmp(TCG_COND_LT, ra, rb, rc, islit, lit);

            break;

        case 0x60:

            /* ADDQ/V */

            gen_arith3(helper_addqv, ra, rb, rc, islit, lit);

            break;

        case 0x69:

            /* SUBQ/V */

            gen_arith3(helper_subqv, ra, rb, rc, islit, lit);

            break;

        case 0x6D:

            /* CMPLE */

            gen_cmp(TCG_COND_LE, ra, rb, rc, islit, lit);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x11:

        switch (fn7) {

        case 0x00:

            /* AND */

            if (likely(rc != 31)) {

                if (ra == 31)

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

                else if (islit)

                    tcg_gen_andi_i64(cpu_ir[rc], cpu_ir[ra], lit);

                else

                    tcg_gen_and_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

            }

            break;

        case 0x08:

            /* BIC */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_andi_i64(cpu_ir[rc], cpu_ir[ra], ~lit);

                    else

                        tcg_gen_andc_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                } else

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

            }

            break;

        case 0x14:

            /* CMOVLBS */

            gen_cmov(TCG_COND_EQ, ra, rb, rc, islit, lit, 1);

            break;

        case 0x16:

            /* CMOVLBC */

            gen_cmov(TCG_COND_NE, ra, rb, rc, islit, lit, 1);

            break;

        case 0x20:

            /* BIS */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_ori_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    else

                        tcg_gen_or_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    else

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x24:

            /* CMOVEQ */

            gen_cmov(TCG_COND_NE, ra, rb, rc, islit, lit, 0);

            break;

        case 0x26:

            /* CMOVNE */

            gen_cmov(TCG_COND_EQ, ra, rb, rc, islit, lit, 0);

            break;

        case 0x28:

            /* ORNOT */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_ori_i64(cpu_ir[rc], cpu_ir[ra], ~lit);

                    else

                        tcg_gen_orc_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], ~lit);

                    else

                        tcg_gen_not_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x40:

            /* XOR */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_xori_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    else

                        tcg_gen_xor_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    else

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x44:

            /* CMOVLT */

            gen_cmov(TCG_COND_GE, ra, rb, rc, islit, lit, 0);

            break;

        case 0x46:

            /* CMOVGE */

            gen_cmov(TCG_COND_LT, ra, rb, rc, islit, lit, 0);

            break;

        case 0x48:

            /* EQV */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_xori_i64(cpu_ir[rc], cpu_ir[ra], ~lit);

                    else

                        tcg_gen_eqv_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                } else {

                    if (islit)

                        tcg_gen_movi_i64(cpu_ir[rc], ~lit);

                    else

                        tcg_gen_not_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x61:

            /* AMASK */

            if (likely(rc != 31)) {

                if (islit)

                    tcg_gen_movi_i64(cpu_ir[rc], helper_amask(lit));

                else

                    tcg_gen_helper_1_1(helper_amask, cpu_ir[rc], cpu_ir[rb]);

            }

            break;

        case 0x64:

            /* CMOVLE */

            gen_cmov(TCG_COND_GT, ra, rb, rc, islit, lit, 0);

            break;

        case 0x66:

            /* CMOVGT */

            gen_cmov(TCG_COND_LE, ra, rb, rc, islit, lit, 0);

            break;

        case 0x6C:

            /* IMPLVER */

            if (rc != 31)

                tcg_gen_helper_1_0(helper_load_implver, cpu_ir[rc]);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x12:

        switch (fn7) {

        case 0x02:

            /* MSKBL */

            gen_arith3(helper_mskbl, ra, rb, rc, islit, lit);

            break;

        case 0x06:

            /* EXTBL */

            gen_ext_l(&tcg_gen_ext8u_i64, ra, rb, rc, islit, lit);

            break;

        case 0x0B:

            /* INSBL */

            gen_arith3(helper_insbl, ra, rb, rc, islit, lit);

            break;

        case 0x12:

            /* MSKWL */

            gen_arith3(helper_mskwl, ra, rb, rc, islit, lit);

            break;

        case 0x16:

            /* EXTWL */

            gen_ext_l(&tcg_gen_ext16u_i64, ra, rb, rc, islit, lit);

            break;

        case 0x1B:

            /* INSWL */

            gen_arith3(helper_inswl, ra, rb, rc, islit, lit);

            break;

        case 0x22:

            /* MSKLL */

            gen_arith3(helper_mskll, ra, rb, rc, islit, lit);

            break;

        case 0x26:

            /* EXTLL */

            gen_ext_l(&tcg_gen_ext32u_i64, ra, rb, rc, islit, lit);

            break;

        case 0x2B:

            /* INSLL */

            gen_arith3(helper_insll, ra, rb, rc, islit, lit);

            break;

        case 0x30:

            /* ZAP */

            gen_arith3(helper_zap, ra, rb, rc, islit, lit);

            break;

        case 0x31:

            /* ZAPNOT */

            gen_arith3(helper_zapnot, ra, rb, rc, islit, lit);

            break;

        case 0x32:

            /* MSKQL */

            gen_arith3(helper_mskql, ra, rb, rc, islit, lit);

            break;

        case 0x34:

            /* SRL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_shri_i64(cpu_ir[rc], cpu_ir[ra], lit & 0x3f);

                    else {

                        TCGv shift = tcg_temp_new(TCG_TYPE_I64);

                        tcg_gen_andi_i64(shift, cpu_ir[rb], 0x3f);

                        tcg_gen_shr_i64(cpu_ir[rc], cpu_ir[ra], shift);

                        tcg_temp_free(shift);

                    }

                } else

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

            }

            break;

        case 0x36:

            /* EXTQL */

            gen_ext_l(NULL, ra, rb, rc, islit, lit);

            break;

        case 0x39:

            /* SLL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_shli_i64(cpu_ir[rc], cpu_ir[ra], lit & 0x3f);

                    else {

                        TCGv shift = tcg_temp_new(TCG_TYPE_I64);

                        tcg_gen_andi_i64(shift, cpu_ir[rb], 0x3f);

                        tcg_gen_shl_i64(cpu_ir[rc], cpu_ir[ra], shift);

                        tcg_temp_free(shift);

                    }

                } else

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

            }

            break;

        case 0x3B:

            /* INSQL */

            gen_arith3(helper_insql, ra, rb, rc, islit, lit);

            break;

        case 0x3C:

            /* SRA */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit)

                        tcg_gen_sari_i64(cpu_ir[rc], cpu_ir[ra], lit & 0x3f);

                    else {

                        TCGv shift = tcg_temp_new(TCG_TYPE_I64);

                        tcg_gen_andi_i64(shift, cpu_ir[rb], 0x3f);

                        tcg_gen_sar_i64(cpu_ir[rc], cpu_ir[ra], shift);

                        tcg_temp_free(shift);

                    }

                } else

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

            }

            break;

        case 0x52:

            /* MSKWH */

            gen_arith3(helper_mskwh, ra, rb, rc, islit, lit);

            break;

        case 0x57:

            /* INSWH */

            gen_arith3(helper_inswh, ra, rb, rc, islit, lit);

            break;

        case 0x5A:

            /* EXTWH */

            gen_ext_h(&tcg_gen_ext16u_i64, ra, rb, rc, islit, lit);

            break;

        case 0x62:

            /* MSKLH */

            gen_arith3(helper_msklh, ra, rb, rc, islit, lit);

            break;

        case 0x67:

            /* INSLH */

            gen_arith3(helper_inslh, ra, rb, rc, islit, lit);

            break;

        case 0x6A:

            /* EXTLH */

            gen_ext_h(&tcg_gen_ext16u_i64, ra, rb, rc, islit, lit);

            break;

        case 0x72:

            /* MSKQH */

            gen_arith3(helper_mskqh, ra, rb, rc, islit, lit);

            break;

        case 0x77:

            /* INSQH */

            gen_arith3(helper_insqh, ra, rb, rc, islit, lit);

            break;

        case 0x7A:

            /* EXTQH */

            gen_ext_h(NULL, ra, rb, rc, islit, lit);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x13:

        switch (fn7) {

        case 0x00:

            /* MULL */

            if (likely(rc != 31)) {

                if (ra == 31)

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

                else {

                    if (islit)

                        tcg_gen_muli_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    else

                        tcg_gen_mul_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                }

            }

            break;

        case 0x20:

            /* MULQ */

            if (likely(rc != 31)) {

                if (ra == 31)

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

                else if (islit)

                    tcg_gen_muli_i64(cpu_ir[rc], cpu_ir[ra], lit);

                else

                    tcg_gen_mul_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

            }

            break;

        case 0x30:

            /* UMULH */

            gen_arith3(helper_umulh, ra, rb, rc, islit, lit);

            break;

        case 0x40:

            /* MULL/V */

            gen_arith3(helper_mullv, ra, rb, rc, islit, lit);

            break;

        case 0x60:

            /* MULQ/V */

            gen_arith3(helper_mulqv, ra, rb, rc, islit, lit);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x14:

        switch (fpfn) { /* f11 & 0x3F */

        case 0x04:

            /* ITOFS */

            if (!(ctx->amask & AMASK_FIX))

                goto invalid_opc;

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I32);

                    tcg_gen_trunc_i64_i32(tmp, cpu_ir[ra]);

                    tcg_gen_helper_1_1(helper_memory_to_s, cpu_fir[rc], tmp);

                    tcg_temp_free(tmp);

                } else

                    tcg_gen_movi_i64(cpu_fir[rc], 0);

            }

            break;

        case 0x0A:

            /* SQRTF */

            if (!(ctx->amask & AMASK_FIX))

                goto invalid_opc;

            gen_farith2(&helper_sqrtf, rb, rc);

            break;

        case 0x0B:

            /* SQRTS */

            if (!(ctx->amask & AMASK_FIX))

                goto invalid_opc;

            gen_farith2(&helper_sqrts, rb, rc);

            break;

        case 0x14:

            /* ITOFF */

            if (!(ctx->amask & AMASK_FIX))

                goto invalid_opc;

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new(TCG_TYPE_I32);

                    tcg_gen_trunc_i64_i32(tmp, cpu_ir[ra]);

                    tcg_gen_helper_1_1(helper_memory_to_f, cpu_fir[rc], tmp);

                    tcg_temp_free(tmp);

                } else

                    tcg_gen_movi_i64(cpu_fir[rc], 0);

            }

            break;

        case 0x24:

            /* ITOFT */

            if (!(ctx->amask & AMASK_FIX))

                goto invalid_opc;

            if (likely(rc != 31)) {

                if (ra != 31)

                    tcg_gen_mov_i64(cpu_fir[rc], cpu_ir[ra]);

                else

                    tcg_gen_movi_i64(cpu_fir[rc], 0);

            }

            break;

        case 0x2A:

            /* SQRTG */

            if (!(ctx->amask & AMASK_FIX))

                goto invalid_opc;

            gen_farith2(&helper_sqrtg, rb, rc);

            break;

        case 0x02B:

            /* SQRTT */

            if (!(ctx->amask & AMASK_FIX))

                goto invalid_opc;

            gen_farith2(&helper_sqrtt, rb, rc);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x15:

        /* VAX floating point */

        /* XXX: rounding mode and trap are ignored (!) */

        switch (fpfn) { /* f11 & 0x3F */

        case 0x00:

            /* ADDF */

            gen_farith3(&helper_addf, ra, rb, rc);

            break;

        case 0x01:

            /* SUBF */

            gen_farith3(&helper_subf, ra, rb, rc);

            break;

        case 0x02:

            /* MULF */

            gen_farith3(&helper_mulf, ra, rb, rc);

            break;

        case 0x03:

            /* DIVF */

            gen_farith3(&helper_divf, ra, rb, rc);

            break;

        case 0x1E:

            /* CVTDG */

#if 0 // TODO

            gen_farith2(&helper_cvtdg, rb, rc);

#else

            goto invalid_opc;

#endif

            break;

        case 0x20:

            /* ADDG */

            gen_farith3(&helper_addg, ra, rb, rc);

            break;

        case 0x21:

            /* SUBG */

            gen_farith3(&helper_subg, ra, rb, rc);

            break;

        case 0x22:

            /* MULG */

            gen_farith3(&helper_mulg, ra, rb, rc);

            break;

        case 0x23:

            /* DIVG */

            gen_farith3(&helper_divg, ra, rb, rc);

            break;

        case 0x25:

            /* CMPGEQ */

            gen_farith3(&helper_cmpgeq, ra, rb, rc);

            break;

        case 0x26:

            /* CMPGLT */

            gen_farith3(&helper_cmpglt, ra, rb, rc);

            break;

        case 0x27:

            /* CMPGLE */

            gen_farith3(&helper_cmpgle, ra, rb, rc);

            break;

        case 0x2C:

            /* CVTGF */

            gen_farith2(&helper_cvtgf, rb, rc);

            break;

        case 0x2D:

            /* CVTGD */

#if 0 // TODO

            gen_farith2(ctx, &helper_cvtgd, rb, rc);

#else

            goto invalid_opc;

#endif

            break;

        case 0x2F:

            /* CVTGQ */

            gen_farith2(&helper_cvtgq, rb, rc);

            break;

        case 0x3C:

            /* CVTQF */

            gen_farith2(&helper_cvtqf, rb, rc);

            break;

        case 0x3E:

            /* CVTQG */

            gen_farith2(&helper_cvtqg, rb, rc);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x16:

        /* IEEE floating-point */

        /* XXX: rounding mode and traps are ignored (!) */

        switch (fpfn) { /* f11 & 0x3F */

        case 0x00:

            /* ADDS */

            gen_farith3(&helper_adds, ra, rb, rc);

            break;

        case 0x01:

            /* SUBS */

            gen_farith3(&helper_subs, ra, rb, rc);

            break;

        case 0x02:

            /* MULS */

            gen_farith3(&helper_muls, ra, rb, rc);

            break;

        case 0x03:

            /* DIVS */

            gen_farith3(&helper_divs, ra, rb, rc);

            break;

        case 0x20:

            /* ADDT */

            gen_farith3(&helper_addt, ra, rb, rc);

            break;

        case 0x21:

            /* SUBT */

            gen_farith3(&helper_subt, ra, rb, rc);

            break;

        case 0x22:

            /* MULT */

            gen_farith3(&helper_mult, ra, rb, rc);

            break;

        case 0x23:

            /* DIVT */

            gen_farith3(&helper_divt, ra, rb, rc);

            break;

        case 0x24:

            /* CMPTUN */

            gen_farith3(&helper_cmptun, ra, rb, rc);

            break;

        case 0x25:

            /* CMPTEQ */

            gen_farith3(&helper_cmpteq, ra, rb, rc);

            break;

        case 0x26:

            /* CMPTLT */

            gen_farith3(&helper_cmptlt, ra, rb, rc);

            break;

        case 0x27:

            /* CMPTLE */

            gen_farith3(&helper_cmptle, ra, rb, rc);

            break;

        case 0x2C:

            /* XXX: incorrect */

            if (fn11 == 0x2AC) {

                /* CVTST */

                gen_farith2(&helper_cvtst, rb, rc);

            } else {

                /* CVTTS */

                gen_farith2(&helper_cvtts, rb, rc);

            }

            break;

        case 0x2F:

            /* CVTTQ */

            gen_farith2(&helper_cvttq, rb, rc);

            break;

        case 0x3C:

            /* CVTQS */

            gen_farith2(&helper_cvtqs, rb, rc);

            break;

        case 0x3E:

            /* CVTQT */

            gen_farith2(&helper_cvtqt, rb, rc);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x17:

        switch (fn11) {

        case 0x010:

            /* CVTLQ */

            gen_farith2(&helper_cvtlq, rb, rc);

            break;

        case 0x020:

            if (likely(rc != 31)) {

                if (ra == rb)

                    /* FMOV */

                    tcg_gen_mov_i64(cpu_fir[rc], cpu_fir[ra]);

                else

                    /* CPYS */

                    gen_farith3(&helper_cpys, ra, rb, rc);

            }

            break;

        case 0x021:

            /* CPYSN */

            gen_farith3(&helper_cpysn, ra, rb, rc);

            break;

        case 0x022:

            /* CPYSE */

            gen_farith3(&helper_cpyse, ra, rb, rc);

            break;

        case 0x024:

            /* MT_FPCR */

            if (likely(ra != 31))

                tcg_gen_helper_0_1(helper_store_fpcr, cpu_fir[ra]);

            else {

                TCGv tmp = tcg_const_i64(0);

                tcg_gen_helper_0_1(helper_store_fpcr, tmp);

                tcg_temp_free(tmp);

            }

            break;

        case 0x025:

            /* MF_FPCR */

            if (likely(ra != 31))

                tcg_gen_helper_1_0(helper_load_fpcr, cpu_fir[ra]);

            break;

        case 0x02A:

            /* FCMOVEQ */

            gen_fcmov(&helper_cmpfeq, ra, rb, rc);

            break;

        case 0x02B:

            /* FCMOVNE */

            gen_fcmov(&helper_cmpfne, ra, rb, rc);

            break;

        case 0x02C:

            /* FCMOVLT */

            gen_fcmov(&helper_cmpflt, ra, rb, rc);

            break;

        case 0x02D:

            /* FCMOVGE */

            gen_fcmov(&helper_cmpfge, ra, rb, rc);

            break;

        case 0x02E:

            /* FCMOVLE */

            gen_fcmov(&helper_cmpfle, ra, rb, rc);

            break;

        case 0x02F:

            /* FCMOVGT */

            gen_fcmov(&helper_cmpfgt, ra, rb, rc);

            break;

        case 0x030:

            /* CVTQL */

            gen_farith2(&helper_cvtql, rb, rc);

            break;

        case 0x130:

            /* CVTQL/V */

            gen_farith2(&helper_cvtqlv, rb, rc);

            break;

        case 0x530:

            /* CVTQL/SV */

            gen_farith2(&helper_cvtqlsv, rb, rc);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x18:

        switch ((uint16_t)disp16) {

        case 0x0000:

            /* TRAPB */

            /* No-op. Just exit from the current tb */

            ret = 2;

            break;

        case 0x0400:

            /* EXCB */

            /* No-op. Just exit from the current tb */

            ret = 2;

            break;

        case 0x4000:

            /* MB */

            /* No-op */

            break;

        case 0x4400:

            /* WMB */

            /* No-op */

            break;

        case 0x8000:

            /* FETCH */

            /* No-op */

            break;

        case 0xA000:

            /* FETCH_M */

            /* No-op */

            break;

        case 0xC000:

            /* RPCC */

            if (ra != 31)

                tcg_gen_helper_1_0(helper_load_pcc, cpu_ir[ra]);

            break;

        case 0xE000:

            /* RC */

            if (ra != 31)

                tcg_gen_helper_1_0(helper_rc, cpu_ir[ra]);

            break;

        case 0xE800:

            /* ECB */

            /* XXX: TODO: evict tb cache at address rb */

#if 0

            ret = 2;

#else

            goto invalid_opc;

#endif

            break;

        case 0xF000:

            /* RS */

            if (ra != 31)

                tcg_gen_helper_1_0(helper_rs, cpu_ir[ra]);

            break;

        case 0xF800:

            /* WH64 */

            /* No-op */

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x19:

        /* HW_MFPR (PALcode) */

#if defined (CONFIG_USER_ONLY)

        goto invalid_opc;

#else

        if (!ctx->pal_mode)

            goto invalid_opc;

        if (ra != 31) {

            TCGv tmp = tcg_const_i32(insn & 0xFF);

            tcg_gen_helper_1_2(helper_mfpr, cpu_ir[ra], tmp, cpu_ir[ra]);

            tcg_temp_free(tmp);

        }

        break;

#endif

    case 0x1A:

        if (ra != 31)

            tcg_gen_movi_i64(cpu_ir[ra], ctx->pc);

        if (rb != 31)

            tcg_gen_andi_i64(cpu_pc, cpu_ir[rb], ~3);

        else

            tcg_gen_movi_i64(cpu_pc, 0);

        /* Those four jumps only differ by the branch prediction hint */

        switch (fn2) {

        case 0x0:

            /* JMP */

            break;

        case 0x1:

            /* JSR */

            break;

        case 0x2:

            /* RET */

            break;

        case 0x3:

            /* JSR_COROUTINE */

            break;

        }

        ret = 1;

        break;

    case 0x1B:

        /* HW_LD (PALcode) */

#if defined (CONFIG_USER_ONLY)

        goto invalid_opc;

#else

        if (!ctx->pal_mode)

            goto invalid_opc;

        if (ra != 31) {

            TCGv addr = tcg_temp_new(TCG_TYPE_I64);

            if (rb != 31)

                tcg_gen_addi_i64(addr, cpu_ir[rb], disp12);

            else

                tcg_gen_movi_i64(addr, disp12);

            switch ((insn >> 12) & 0xF) {

            case 0x0:

                /* Longword physical access */

                tcg_gen_helper_0_2(helper_ldl_raw, cpu_ir[ra], addr);

                break;

            case 0x1:

                /* Quadword physical access */

                tcg_gen_helper_0_2(helper_ldq_raw, cpu_ir[ra], addr);

                break;

            case 0x2:

                /* Longword physical access with lock */

                tcg_gen_helper_0_2(helper_ldl_l_raw, cpu_ir[ra], addr);

                break;

            case 0x3:

                /* Quadword physical access with lock */

                tcg_gen_helper_0_2(helper_ldq_l_raw, cpu_ir[ra], addr);

                break;

            case 0x4:

                /* Longword virtual PTE fetch */

                tcg_gen_helper_0_2(helper_ldl_kernel, cpu_ir[ra], addr);

                break;

            case 0x5:

                /* Quadword virtual PTE fetch */

                tcg_gen_helper_0_2(helper_ldq_kernel, cpu_ir[ra], addr);

                break;

            case 0x6:

                /* Incpu_ir[ra]id */

                goto incpu_ir[ra]id_opc;

            case 0x7:

                /* Incpu_ir[ra]id */

                goto incpu_ir[ra]id_opc;

            case 0x8:

                /* Longword virtual access */

                tcg_gen_helper_1_1(helper_st_virt_to_phys, addr, addr);

                tcg_gen_helper_0_2(helper_ldl_raw, cpu_ir[ra], addr);

                break;

            case 0x9:

                /* Quadword virtual access */

                tcg_gen_helper_1_1(helper_st_virt_to_phys, addr, addr);

                tcg_gen_helper_0_2(helper_ldq_raw, cpu_ir[ra], addr);

                break;

            case 0xA:

                /* Longword virtual access with protection check */

                tcg_gen_qemu_ld32s(cpu_ir[ra], addr, ctx->flags);

                break;

            case 0xB:

                /* Quadword virtual access with protection check */

                tcg_gen_qemu_ld64(cpu_ir[ra], addr, ctx->flags);

                break;

            case 0xC:

                /* Longword virtual access with altenate access mode */

                tcg_gen_helper_0_0(helper_set_alt_mode);

                tcg_gen_helper_1_1(helper_st_virt_to_phys, addr, addr);

                tcg_gen_helper_0_2(helper_ldl_raw, cpu_ir[ra], addr);

                tcg_gen_helper_0_0(helper_restore_mode);

                break;

            case 0xD:

                /* Quadword virtual access with altenate access mode */

                tcg_gen_helper_0_0(helper_set_alt_mode);

                tcg_gen_helper_1_1(helper_st_virt_to_phys, addr, addr);

                tcg_gen_helper_0_2(helper_ldq_raw, cpu_ir[ra], addr);

                tcg_gen_helper_0_0(helper_restore_mode);

                break;

            case 0xE:

                /* Longword virtual access with alternate access mode and

                 * protection checks

                 */

                tcg_gen_helper_0_0(helper_set_alt_mode);

                tcg_gen_helper_0_2(helper_ldl_data, cpu_ir[ra], addr);

                tcg_gen_helper_0_0(helper_restore_mode);

                break;

            case 0xF:

                /* Quadword virtual access with alternate access mode and

                 * protection checks

                 */

                tcg_gen_helper_0_0(helper_set_alt_mode);

                tcg_gen_helper_0_2(helper_ldq_data, cpu_ir[ra], addr);

                tcg_gen_helper_0_0(helper_restore_mode);

                break;

            }

            tcg_temp_free(addr);

        }

        break;

#endif

    case 0x1C:

        switch (fn7) {

        case 0x00:

            /* SEXTB */

            if (!(ctx->amask & AMASK_BWX))

                goto invalid_opc;

            if (likely(rc != 31)) {

                if (islit)

                    tcg_gen_movi_i64(cpu_ir[rc], (int64_t)((int8_t)lit));

                else

                    tcg_gen_ext8s_i64(cpu_ir[rc], cpu_ir[rb]);

            }

            break;

        case 0x01:

            /* SEXTW */

            if (!(ctx->amask & AMASK_BWX))

                goto invalid_opc;

            if (likely(rc != 31)) {

                if (islit)

                    tcg_gen_movi_i64(cpu_ir[rc], (int64_t)((int16_t)lit));

                else

                    tcg_gen_ext16s_i64(cpu_ir[rc], cpu_ir[rb]);

            }

            break;

        case 0x30:

            /* CTPOP */

            if (!(ctx->amask & AMASK_CIX))

                goto invalid_opc;

            if (likely(rc != 31)) {

                if (islit)

                    tcg_gen_movi_i64(cpu_ir[rc], ctpop64(lit));

                else

                    tcg_gen_helper_1_1(helper_ctpop, cpu_ir[rc], cpu_ir[rb]);

            }

            break;

        case 0x31:

            /* PERR */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x32:

            /* CTLZ */

            if (!(ctx->amask & AMASK_CIX))

                goto invalid_opc;

            if (likely(rc != 31)) {

                if (islit)

                    tcg_gen_movi_i64(cpu_ir[rc], clz64(lit));

                else

                    tcg_gen_helper_1_1(helper_ctlz, cpu_ir[rc], cpu_ir[rb]);

            }

            break;

        case 0x33:

            /* CTTZ */

            if (!(ctx->amask & AMASK_CIX))

                goto invalid_opc;

            if (likely(rc != 31)) {

                if (islit)

                    tcg_gen_movi_i64(cpu_ir[rc], ctz64(lit));

                else

                    tcg_gen_helper_1_1(helper_cttz, cpu_ir[rc], cpu_ir[rb]);

            }

            break;

        case 0x34:

            /* UNPKBW */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x35:

            /* UNPKWL */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x36:

            /* PKWB */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x37:

            /* PKLB */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x38:

            /* MINSB8 */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x39:

            /* MINSW4 */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x3A:

            /* MINUB8 */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x3B:

            /* MINUW4 */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x3C:

            /* MAXUB8 */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x3D:

            /* MAXUW4 */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x3E:

            /* MAXSB8 */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x3F:

            /* MAXSW4 */

            if (!(ctx->amask & AMASK_MVI))

                goto invalid_opc;

            /* XXX: TODO */

            goto invalid_opc;

            break;

        case 0x70:

            /* FTOIT */

            if (!(ctx->amask & AMASK_FIX))

                goto invalid_opc;

            if (likely(rc != 31)) {

                if (ra != 31)

                    tcg_gen_mov_i64(cpu_ir[rc], cpu_fir[ra]);

                else

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

            }

            break;

        case 0x78:

            /* FTOIS */

            if (!(ctx->amask & AMASK_FIX))

                goto invalid_opc;

            if (rc != 31) {

                TCGv tmp1 = tcg_temp_new(TCG_TYPE_I32);

                if (ra != 31)

                    tcg_gen_helper_1_1(helper_s_to_memory, tmp1, cpu_fir[ra]);

                else {

                    TCGv tmp2 = tcg_const_i64(0);

                    tcg_gen_helper_1_1(helper_s_to_memory, tmp1, tmp2);

                    tcg_temp_free(tmp2);

                }

                tcg_gen_ext_i32_i64(cpu_ir[rc], tmp1);

                tcg_temp_free(tmp1);

            }

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x1D:

        /* HW_MTPR (PALcode) */

#if defined (CONFIG_USER_ONLY)

        goto invalid_opc;

#else

        if (!ctx->pal_mode)

            goto invalid_opc;

        else {

            TCGv tmp1 = tcg_const_i32(insn & 0xFF);

            if (ra != 31)

                tcg_gen_helper(helper_mtpr, tmp1, cpu_ir[ra]);

            else {

                TCGv tmp2 = tcg_const_i64(0);

                tcg_gen_helper(helper_mtpr, tmp1, tmp2);

                tcg_temp_free(tmp2);

            }

            tcg_temp_free(tmp1);

            ret = 2;

        }

        break;

#endif

    case 0x1E:

        /* HW_REI (PALcode) */

#if defined (CONFIG_USER_ONLY)

        goto invalid_opc;

#else

        if (!ctx->pal_mode)

            goto invalid_opc;

        if (rb == 31) {

            /* "Old" alpha */

            tcg_gen_helper_0_0(helper_hw_rei);

        } else {

            TCGv tmp;



            if (ra != 31) {

                tmp = tcg_temp_new(TCG_TYPE_I64);

                tcg_gen_addi_i64(tmp, cpu_ir[rb], (((int64_t)insn << 51) >> 51));

            } else

                tmp = tcg_const_i64(((int64_t)insn << 51) >> 51);

            tcg_gen_helper_0_1(helper_hw_ret, tmp);

            tcg_temp_free(tmp);

        }

        ret = 2;

        break;

#endif

    case 0x1F:

        /* HW_ST (PALcode) */

#if defined (CONFIG_USER_ONLY)

        goto invalid_opc;

#else

        if (!ctx->pal_mode)

            goto invalid_opc;

        else {

            TCGv addr, val;

            addr = tcg_temp_new(TCG_TYPE_I64);

            if (rb != 31)

                tcg_gen_addi_i64(addr, cpu_ir[rb], disp12);

            else

                tcg_gen_movi_i64(addr, disp12);

            if (ra != 31)

                val = cpu_ir[ra];

            else {

                val = tcg_temp_new(TCG_TYPE_I64);

                tcg_gen_movi_i64(val, 0);

            }

            switch ((insn >> 12) & 0xF) {

            case 0x0:

                /* Longword physical access */

                tcg_gen_helper_0_2(helper_stl_raw, val, addr);

                break;

            case 0x1:

                /* Quadword physical access */

                tcg_gen_helper_0_2(helper_stq_raw, val, addr);

                break;

            case 0x2:

                /* Longword physical access with lock */

                tcg_gen_helper_1_2(helper_stl_c_raw, val, val, addr);

                break;

            case 0x3:

                /* Quadword physical access with lock */

                tcg_gen_helper_1_2(helper_stq_c_raw, val, val, addr);

                break;

            case 0x4:

                /* Longword virtual access */

                tcg_gen_helper_1_1(helper_st_virt_to_phys, addr, addr);

                tcg_gen_helper_0_2(helper_stl_raw, val, addr);

                break;

            case 0x5:

                /* Quadword virtual access */

                tcg_gen_helper_1_1(helper_st_virt_to_phys, addr, addr);

                tcg_gen_helper_0_2(helper_stq_raw, val, addr);

                break;

            case 0x6:

                /* Invalid */

                goto invalid_opc;

            case 0x7:

                /* Invalid */

                goto invalid_opc;

            case 0x8:

                /* Invalid */

                goto invalid_opc;

            case 0x9:

                /* Invalid */

                goto invalid_opc;

            case 0xA:

                /* Invalid */

                goto invalid_opc;

            case 0xB:

                /* Invalid */

                goto invalid_opc;

            case 0xC:

                /* Longword virtual access with alternate access mode */

                tcg_gen_helper_0_0(helper_set_alt_mode);

                tcg_gen_helper_1_1(helper_st_virt_to_phys, addr, addr);

                tcg_gen_helper_0_2(helper_stl_raw, val, addr);

                tcg_gen_helper_0_0(helper_restore_mode);

                break;

            case 0xD:

                /* Quadword virtual access with alternate access mode */

                tcg_gen_helper_0_0(helper_set_alt_mode);

                tcg_gen_helper_1_1(helper_st_virt_to_phys, addr, addr);

                tcg_gen_helper_0_2(helper_stl_raw, val, addr);

                tcg_gen_helper_0_0(helper_restore_mode);

                break;

            case 0xE:

                /* Invalid */

                goto invalid_opc;

            case 0xF:

                /* Invalid */

                goto invalid_opc;

            }

            if (ra != 31)

                tcg_temp_free(val);

            tcg_temp_free(addr);

        }

        ret = 2;

        break;

#endif

    case 0x20:

        /* LDF */

        gen_load_mem(ctx, &gen_qemu_ldf, ra, rb, disp16, 1, 0);

        break;

    case 0x21:

        /* LDG */

        gen_load_mem(ctx, &gen_qemu_ldg, ra, rb, disp16, 1, 0);

        break;

    case 0x22:

        /* LDS */

        gen_load_mem(ctx, &gen_qemu_lds, ra, rb, disp16, 1, 0);

        break;

    case 0x23:

        /* LDT */

        gen_load_mem(ctx, &tcg_gen_qemu_ld64, ra, rb, disp16, 1, 0);

        break;

    case 0x24:

        /* STF */

        gen_store_mem(ctx, &gen_qemu_stf, ra, rb, disp16, 1, 0);

        break;

    case 0x25:

        /* STG */

        gen_store_mem(ctx, &gen_qemu_stg, ra, rb, disp16, 1, 0);

        break;

    case 0x26:

        /* STS */

        gen_store_mem(ctx, &gen_qemu_sts, ra, rb, disp16, 1, 0);

        break;

    case 0x27:

        /* STT */

        gen_store_mem(ctx, &tcg_gen_qemu_st64, ra, rb, disp16, 1, 0);

        break;

    case 0x28:

        /* LDL */

        gen_load_mem(ctx, &tcg_gen_qemu_ld32s, ra, rb, disp16, 0, 0);

        break;

    case 0x29:

        /* LDQ */

        gen_load_mem(ctx, &tcg_gen_qemu_ld64, ra, rb, disp16, 0, 0);

        break;

    case 0x2A:

        /* LDL_L */

        gen_load_mem(ctx, &gen_qemu_ldl_l, ra, rb, disp16, 0, 0);

        break;

    case 0x2B:

        /* LDQ_L */

        gen_load_mem(ctx, &gen_qemu_ldq_l, ra, rb, disp16, 0, 0);

        break;

    case 0x2C:

        /* STL */

        gen_store_mem(ctx, &tcg_gen_qemu_st32, ra, rb, disp16, 0, 0);

        break;

    case 0x2D:

        /* STQ */

        gen_store_mem(ctx, &tcg_gen_qemu_st64, ra, rb, disp16, 0, 0);

        break;

    case 0x2E:

        /* STL_C */

        gen_store_mem(ctx, &gen_qemu_stl_c, ra, rb, disp16, 0, 0);

        break;

    case 0x2F:

        /* STQ_C */

        gen_store_mem(ctx, &gen_qemu_stq_c, ra, rb, disp16, 0, 0);

        break;

    case 0x30:

        /* BR */

        if (ra != 31)

            tcg_gen_movi_i64(cpu_ir[ra], ctx->pc);

        tcg_gen_movi_i64(cpu_pc, ctx->pc + (int64_t)(disp21 << 2));

        ret = 1;

        break;

    case 0x31:

        /* FBEQ */

        gen_fbcond(ctx, &helper_cmpfeq, ra, disp16);

        ret = 1;

        break;

    case 0x32:

        /* FBLT */

        gen_fbcond(ctx, &helper_cmpflt, ra, disp16);

        ret = 1;

        break;

    case 0x33:

        /* FBLE */

        gen_fbcond(ctx, &helper_cmpfle, ra, disp16);

        ret = 1;

        break;

    case 0x34:

        /* BSR */

        if (ra != 31)

            tcg_gen_movi_i64(cpu_ir[ra], ctx->pc);

        tcg_gen_movi_i64(cpu_pc, ctx->pc + (int64_t)(disp21 << 2));

        ret = 1;

        break;

    case 0x35:

        /* FBNE */

        gen_fbcond(ctx, &helper_cmpfne, ra, disp16);

        ret = 1;

        break;

    case 0x36:

        /* FBGE */

        gen_fbcond(ctx, &helper_cmpfge, ra, disp16);

        ret = 1;

        break;

    case 0x37:

        /* FBGT */

        gen_fbcond(ctx, &helper_cmpfgt, ra, disp16);

        ret = 1;

        break;

    case 0x38:

        /* BLBC */

        gen_bcond(ctx, TCG_COND_EQ, ra, disp16, 1);

        ret = 1;

        break;

    case 0x39:

        /* BEQ */

        gen_bcond(ctx, TCG_COND_EQ, ra, disp16, 0);

        ret = 1;

        break;

    case 0x3A:

        /* BLT */

        gen_bcond(ctx, TCG_COND_LT, ra, disp16, 0);

        ret = 1;

        break;

    case 0x3B:

        /* BLE */

        gen_bcond(ctx, TCG_COND_LE, ra, disp16, 0);

        ret = 1;

        break;

    case 0x3C:

        /* BLBS */

        gen_bcond(ctx, TCG_COND_NE, ra, disp16, 1);

        ret = 1;

        break;

    case 0x3D:

        /* BNE */

        gen_bcond(ctx, TCG_COND_NE, ra, disp16, 0);

        ret = 1;

        break;

    case 0x3E:

        /* BGE */

        gen_bcond(ctx, TCG_COND_GE, ra, disp16, 0);

        ret = 1;

        break;

    case 0x3F:

        /* BGT */

        gen_bcond(ctx, TCG_COND_GT, ra, disp16, 0);

        ret = 1;

        break;

    invalid_opc:

        gen_invalid(ctx);

        ret = 3;

        break;

    }



    return ret;

}
