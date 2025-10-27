static ExitStatus translate_one(DisasContext *ctx, uint32_t insn)

{

    uint32_t palcode;

    int32_t disp21, disp16;

#ifndef CONFIG_USER_ONLY

    int32_t disp12;

#endif

    uint16_t fn11;

    uint8_t opc, ra, rb, rc, fpfn, fn7, islit, real_islit;

    uint8_t lit;

    ExitStatus ret;



    /* Decode all instruction fields */

    opc = insn >> 26;

    ra = (insn >> 21) & 0x1F;

    rb = (insn >> 16) & 0x1F;

    rc = insn & 0x1F;

    real_islit = islit = (insn >> 12) & 1;

    if (rb == 31 && !islit) {

        islit = 1;

        lit = 0;

    } else

        lit = (insn >> 13) & 0xFF;

    palcode = insn & 0x03FFFFFF;

    disp21 = ((int32_t)((insn & 0x001FFFFF) << 11)) >> 11;

    disp16 = (int16_t)(insn & 0x0000FFFF);

#ifndef CONFIG_USER_ONLY

    disp12 = (int32_t)((insn & 0x00000FFF) << 20) >> 20;

#endif

    fn11 = (insn >> 5) & 0x000007FF;

    fpfn = fn11 & 0x3F;

    fn7 = (insn >> 5) & 0x0000007F;

    LOG_DISAS("opc %02x ra %2d rb %2d rc %2d disp16 %6d\n",

              opc, ra, rb, rc, disp16);



    ret = NO_EXIT;

    switch (opc) {

    case 0x00:

        /* CALL_PAL */

        ret = gen_call_pal(ctx, palcode);

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

            if (rb != 31) {

                tcg_gen_addi_i64(cpu_ir[ra], cpu_ir[rb], disp16);

            } else {

                tcg_gen_movi_i64(cpu_ir[ra], disp16);

            }

        }

        break;

    case 0x09:

        /* LDAH */

        if (likely(ra != 31)) {

            if (rb != 31) {

                tcg_gen_addi_i64(cpu_ir[ra], cpu_ir[rb], disp16 << 16);

            } else {

                tcg_gen_movi_i64(cpu_ir[ra], disp16 << 16);

            }

        }

        break;

    case 0x0A:

        /* LDBU */

        if (ctx->tb->flags & TB_FLAGS_AMASK_BWX) {

            gen_load_mem(ctx, &tcg_gen_qemu_ld8u, ra, rb, disp16, 0, 0);

            break;

        }

        goto invalid_opc;

    case 0x0B:

        /* LDQ_U */

        gen_load_mem(ctx, &tcg_gen_qemu_ld64, ra, rb, disp16, 0, 1);

        break;

    case 0x0C:

        /* LDWU */

        if (ctx->tb->flags & TB_FLAGS_AMASK_BWX) {

            gen_load_mem(ctx, &tcg_gen_qemu_ld16u, ra, rb, disp16, 0, 0);

            break;

        }

        goto invalid_opc;

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

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    } else {

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x02:

            /* S4ADDL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new();

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 2);

                    if (islit) {

                        tcg_gen_addi_i64(tmp, tmp, lit);

                    } else {

                        tcg_gen_add_i64(tmp, tmp, cpu_ir[rb]);

                    }

                    tcg_gen_ext32s_i64(cpu_ir[rc], tmp);

                    tcg_temp_free(tmp);

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    } else {

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x09:

            /* SUBL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_subi_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    } else {

                        tcg_gen_sub_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    }

                    tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    } else {

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                    }

                }

            }

            break;

        case 0x0B:

            /* S4SUBL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new();

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 2);

                    if (islit) {

                        tcg_gen_subi_i64(tmp, tmp, lit);

                    } else {

                        tcg_gen_sub_i64(tmp, tmp, cpu_ir[rb]);

                    }

                    tcg_gen_ext32s_i64(cpu_ir[rc], tmp);

                    tcg_temp_free(tmp);

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    } else {

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                    }

                }

            }

            break;

        case 0x0F:

            /* CMPBGE */

            gen_cmpbge(ra, rb, rc, islit, lit);

            break;

        case 0x12:

            /* S8ADDL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new();

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 3);

                    if (islit) {

                        tcg_gen_addi_i64(tmp, tmp, lit);

                    } else {

                        tcg_gen_add_i64(tmp, tmp, cpu_ir[rb]);

                    }

                    tcg_gen_ext32s_i64(cpu_ir[rc], tmp);

                    tcg_temp_free(tmp);

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    } else {

                        tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x1B:

            /* S8SUBL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new();

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 3);

                    if (islit) {

                        tcg_gen_subi_i64(tmp, tmp, lit);

                    } else {

                       tcg_gen_sub_i64(tmp, tmp, cpu_ir[rb]);

                    }

                    tcg_gen_ext32s_i64(cpu_ir[rc], tmp);

                    tcg_temp_free(tmp);

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    } else {

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

                    if (islit) {

                        tcg_gen_addi_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    } else {

                        tcg_gen_add_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    }

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    } else {

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x22:

            /* S4ADDQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new();

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 2);

                    if (islit) {

                        tcg_gen_addi_i64(cpu_ir[rc], tmp, lit);

                    } else {

                        tcg_gen_add_i64(cpu_ir[rc], tmp, cpu_ir[rb]);

                    }

                    tcg_temp_free(tmp);

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    } else {

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x29:

            /* SUBQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_subi_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    } else {

                        tcg_gen_sub_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    }

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    } else {

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x2B:

            /* S4SUBQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new();

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 2);

                    if (islit) {

                        tcg_gen_subi_i64(cpu_ir[rc], tmp, lit);

                    } else {

                        tcg_gen_sub_i64(cpu_ir[rc], tmp, cpu_ir[rb]);

                    }

                    tcg_temp_free(tmp);

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    } else {

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

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

                    TCGv tmp = tcg_temp_new();

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 3);

                    if (islit) {

                        tcg_gen_addi_i64(cpu_ir[rc], tmp, lit);

                    } else {

                        tcg_gen_add_i64(cpu_ir[rc], tmp, cpu_ir[rb]);

                    }

                    tcg_temp_free(tmp);

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    } else {

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x3B:

            /* S8SUBQ */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv tmp = tcg_temp_new();

                    tcg_gen_shli_i64(tmp, cpu_ir[ra], 3);

                    if (islit) {

                        tcg_gen_subi_i64(cpu_ir[rc], tmp, lit);

                    } else {

                        tcg_gen_sub_i64(cpu_ir[rc], tmp, cpu_ir[rb]);

                    }

                    tcg_temp_free(tmp);

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], -lit);

                    } else {

                        tcg_gen_neg_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x3D:

            /* CMPULE */

            gen_cmp(TCG_COND_LEU, ra, rb, rc, islit, lit);

            break;

        case 0x40:

            /* ADDL/V */

            gen_addlv(ra, rb, rc, islit, lit);

            break;

        case 0x49:

            /* SUBL/V */

            gen_sublv(ra, rb, rc, islit, lit);

            break;

        case 0x4D:

            /* CMPLT */

            gen_cmp(TCG_COND_LT, ra, rb, rc, islit, lit);

            break;

        case 0x60:

            /* ADDQ/V */

            gen_addqv(ra, rb, rc, islit, lit);

            break;

        case 0x69:

            /* SUBQ/V */

            gen_subqv(ra, rb, rc, islit, lit);

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

                if (ra == 31) {

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

                } else if (islit) {

                    tcg_gen_andi_i64(cpu_ir[rc], cpu_ir[ra], lit);

                } else {

                    tcg_gen_and_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                }

            }

            break;

        case 0x08:

            /* BIC */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_andi_i64(cpu_ir[rc], cpu_ir[ra], ~lit);

                    } else {

                        tcg_gen_andc_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    }

                } else

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

            }

            break;

        case 0x14:

            /* CMOVLBS */

            gen_cmov(TCG_COND_NE, ra, rb, rc, islit, lit, 1);

            break;

        case 0x16:

            /* CMOVLBC */

            gen_cmov(TCG_COND_EQ, ra, rb, rc, islit, lit, 1);

            break;

        case 0x20:

            /* BIS */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_ori_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    } else {

                        tcg_gen_or_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    }

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    } else {

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x24:

            /* CMOVEQ */

            gen_cmov(TCG_COND_EQ, ra, rb, rc, islit, lit, 0);

            break;

        case 0x26:

            /* CMOVNE */

            gen_cmov(TCG_COND_NE, ra, rb, rc, islit, lit, 0);

            break;

        case 0x28:

            /* ORNOT */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_ori_i64(cpu_ir[rc], cpu_ir[ra], ~lit);

                    } else {

                        tcg_gen_orc_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    }

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], ~lit);

                    } else {

                        tcg_gen_not_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x40:

            /* XOR */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_xori_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    } else {

                        tcg_gen_xor_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    }

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], lit);

                    } else {

                        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x44:

            /* CMOVLT */

            gen_cmov(TCG_COND_LT, ra, rb, rc, islit, lit, 0);

            break;

        case 0x46:

            /* CMOVGE */

            gen_cmov(TCG_COND_GE, ra, rb, rc, islit, lit, 0);

            break;

        case 0x48:

            /* EQV */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_xori_i64(cpu_ir[rc], cpu_ir[ra], ~lit);

                    } else {

                        tcg_gen_eqv_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    }

                } else {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], ~lit);

                    } else {

                        tcg_gen_not_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

            }

            break;

        case 0x61:

            /* AMASK */

            if (likely(rc != 31)) {

                uint64_t amask = ctx->tb->flags >> TB_FLAGS_AMASK_SHIFT;



                if (islit) {

                    tcg_gen_movi_i64(cpu_ir[rc], lit & ~amask);

                } else {

                    tcg_gen_andi_i64(cpu_ir[rc], cpu_ir[rb], ~amask);

                }

            }

            break;

        case 0x64:

            /* CMOVLE */

            gen_cmov(TCG_COND_LE, ra, rb, rc, islit, lit, 0);

            break;

        case 0x66:

            /* CMOVGT */

            gen_cmov(TCG_COND_GT, ra, rb, rc, islit, lit, 0);

            break;

        case 0x6C:

            /* IMPLVER */

            if (rc != 31) {

                tcg_gen_movi_i64(cpu_ir[rc], ctx->implver);

            }

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x12:

        switch (fn7) {

        case 0x02:

            /* MSKBL */

            gen_msk_l(ra, rb, rc, islit, lit, 0x01);

            break;

        case 0x06:

            /* EXTBL */

            gen_ext_l(ra, rb, rc, islit, lit, 0x01);

            break;

        case 0x0B:

            /* INSBL */

            gen_ins_l(ra, rb, rc, islit, lit, 0x01);

            break;

        case 0x12:

            /* MSKWL */

            gen_msk_l(ra, rb, rc, islit, lit, 0x03);

            break;

        case 0x16:

            /* EXTWL */

            gen_ext_l(ra, rb, rc, islit, lit, 0x03);

            break;

        case 0x1B:

            /* INSWL */

            gen_ins_l(ra, rb, rc, islit, lit, 0x03);

            break;

        case 0x22:

            /* MSKLL */

            gen_msk_l(ra, rb, rc, islit, lit, 0x0f);

            break;

        case 0x26:

            /* EXTLL */

            gen_ext_l(ra, rb, rc, islit, lit, 0x0f);

            break;

        case 0x2B:

            /* INSLL */

            gen_ins_l(ra, rb, rc, islit, lit, 0x0f);

            break;

        case 0x30:

            /* ZAP */

            gen_zap(ra, rb, rc, islit, lit);

            break;

        case 0x31:

            /* ZAPNOT */

            gen_zapnot(ra, rb, rc, islit, lit);

            break;

        case 0x32:

            /* MSKQL */

            gen_msk_l(ra, rb, rc, islit, lit, 0xff);

            break;

        case 0x34:

            /* SRL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_shri_i64(cpu_ir[rc], cpu_ir[ra], lit & 0x3f);

                    } else {

                        TCGv shift = tcg_temp_new();

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

            gen_ext_l(ra, rb, rc, islit, lit, 0xff);

            break;

        case 0x39:

            /* SLL */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_shli_i64(cpu_ir[rc], cpu_ir[ra], lit & 0x3f);

                    } else {

                        TCGv shift = tcg_temp_new();

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

            gen_ins_l(ra, rb, rc, islit, lit, 0xff);

            break;

        case 0x3C:

            /* SRA */

            if (likely(rc != 31)) {

                if (ra != 31) {

                    if (islit) {

                        tcg_gen_sari_i64(cpu_ir[rc], cpu_ir[ra], lit & 0x3f);

                    } else {

                        TCGv shift = tcg_temp_new();

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

            gen_msk_h(ra, rb, rc, islit, lit, 0x03);

            break;

        case 0x57:

            /* INSWH */

            gen_ins_h(ra, rb, rc, islit, lit, 0x03);

            break;

        case 0x5A:

            /* EXTWH */

            gen_ext_h(ra, rb, rc, islit, lit, 0x03);

            break;

        case 0x62:

            /* MSKLH */

            gen_msk_h(ra, rb, rc, islit, lit, 0x0f);

            break;

        case 0x67:

            /* INSLH */

            gen_ins_h(ra, rb, rc, islit, lit, 0x0f);

            break;

        case 0x6A:

            /* EXTLH */

            gen_ext_h(ra, rb, rc, islit, lit, 0x0f);

            break;

        case 0x72:

            /* MSKQH */

            gen_msk_h(ra, rb, rc, islit, lit, 0xff);

            break;

        case 0x77:

            /* INSQH */

            gen_ins_h(ra, rb, rc, islit, lit, 0xff);

            break;

        case 0x7A:

            /* EXTQH */

            gen_ext_h(ra, rb, rc, islit, lit, 0xff);

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

                if (ra == 31) {

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

                } else {

                    if (islit) {

                        tcg_gen_muli_i64(cpu_ir[rc], cpu_ir[ra], lit);

                    } else {

                        tcg_gen_mul_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                    }

                    tcg_gen_ext32s_i64(cpu_ir[rc], cpu_ir[rc]);

                }

            }

            break;

        case 0x20:

            /* MULQ */

            if (likely(rc != 31)) {

                if (ra == 31) {

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

                } else if (islit) {

                    tcg_gen_muli_i64(cpu_ir[rc], cpu_ir[ra], lit);

                } else {

                    tcg_gen_mul_i64(cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                }

            }

            break;

        case 0x30:

            /* UMULH */

            {

                TCGv low;

                if (unlikely(rc == 31)){

                    break;

                }

                if (ra == 31) {

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

                    break;

                }

                low = tcg_temp_new();

                if (islit) {

                    tcg_gen_movi_tl(low, lit);

                    tcg_gen_mulu2_i64(low, cpu_ir[rc], cpu_ir[ra], low);

                } else {

                    tcg_gen_mulu2_i64(low, cpu_ir[rc], cpu_ir[ra], cpu_ir[rb]);

                }

                tcg_temp_free(low);

            }

            break;

        case 0x40:

            /* MULL/V */

            gen_mullv(ra, rb, rc, islit, lit);

            break;

        case 0x60:

            /* MULQ/V */

            gen_mulqv(ra, rb, rc, islit, lit);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x14:

        switch (fpfn) { /* fn11 & 0x3F */

        case 0x04:

            /* ITOFS */

            if ((ctx->tb->flags & TB_FLAGS_AMASK_FIX) == 0) {

                goto invalid_opc;

            }

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv_i32 tmp = tcg_temp_new_i32();

                    tcg_gen_trunc_i64_i32(tmp, cpu_ir[ra]);

                    gen_helper_memory_to_s(cpu_fir[rc], tmp);

                    tcg_temp_free_i32(tmp);

                } else

                    tcg_gen_movi_i64(cpu_fir[rc], 0);

            }

            break;

        case 0x0A:

            /* SQRTF */

            if (ctx->tb->flags & TB_FLAGS_AMASK_FIX) {

                gen_fsqrtf(rb, rc);

                break;

            }

            goto invalid_opc;

        case 0x0B:

            /* SQRTS */

            if (ctx->tb->flags & TB_FLAGS_AMASK_FIX) {

                gen_fsqrts(ctx, rb, rc, fn11);

                break;

            }

            goto invalid_opc;

        case 0x14:

            /* ITOFF */

            if ((ctx->tb->flags & TB_FLAGS_AMASK_FIX) == 0) {

                goto invalid_opc;

            }

            if (likely(rc != 31)) {

                if (ra != 31) {

                    TCGv_i32 tmp = tcg_temp_new_i32();

                    tcg_gen_trunc_i64_i32(tmp, cpu_ir[ra]);

                    gen_helper_memory_to_f(cpu_fir[rc], tmp);

                    tcg_temp_free_i32(tmp);

                } else

                    tcg_gen_movi_i64(cpu_fir[rc], 0);

            }

            break;

        case 0x24:

            /* ITOFT */

            if ((ctx->tb->flags & TB_FLAGS_AMASK_FIX) == 0) {

                goto invalid_opc;

            }

            if (likely(rc != 31)) {

                if (ra != 31) {

                    tcg_gen_mov_i64(cpu_fir[rc], cpu_ir[ra]);

                } else {

                    tcg_gen_movi_i64(cpu_fir[rc], 0);

                }

            }

            break;

        case 0x2A:

            /* SQRTG */

            if (ctx->tb->flags & TB_FLAGS_AMASK_FIX) {

                gen_fsqrtg(rb, rc);

                break;

            }

            goto invalid_opc;

        case 0x02B:

            /* SQRTT */

            if (ctx->tb->flags & TB_FLAGS_AMASK_FIX) {

                gen_fsqrtt(ctx, rb, rc, fn11);

                break;

            }

            goto invalid_opc;

        default:

            goto invalid_opc;

        }

        break;

    case 0x15:

        /* VAX floating point */

        /* XXX: rounding mode and trap are ignored (!) */

        switch (fpfn) { /* fn11 & 0x3F */

        case 0x00:

            /* ADDF */

            gen_faddf(ra, rb, rc);

            break;

        case 0x01:

            /* SUBF */

            gen_fsubf(ra, rb, rc);

            break;

        case 0x02:

            /* MULF */

            gen_fmulf(ra, rb, rc);

            break;

        case 0x03:

            /* DIVF */

            gen_fdivf(ra, rb, rc);

            break;

        case 0x1E:

            /* CVTDG */

#if 0 // TODO

            gen_fcvtdg(rb, rc);

#else

            goto invalid_opc;

#endif

            break;

        case 0x20:

            /* ADDG */

            gen_faddg(ra, rb, rc);

            break;

        case 0x21:

            /* SUBG */

            gen_fsubg(ra, rb, rc);

            break;

        case 0x22:

            /* MULG */

            gen_fmulg(ra, rb, rc);

            break;

        case 0x23:

            /* DIVG */

            gen_fdivg(ra, rb, rc);

            break;

        case 0x25:

            /* CMPGEQ */

            gen_fcmpgeq(ra, rb, rc);

            break;

        case 0x26:

            /* CMPGLT */

            gen_fcmpglt(ra, rb, rc);

            break;

        case 0x27:

            /* CMPGLE */

            gen_fcmpgle(ra, rb, rc);

            break;

        case 0x2C:

            /* CVTGF */

            gen_fcvtgf(rb, rc);

            break;

        case 0x2D:

            /* CVTGD */

#if 0 // TODO

            gen_fcvtgd(rb, rc);

#else

            goto invalid_opc;

#endif

            break;

        case 0x2F:

            /* CVTGQ */

            gen_fcvtgq(rb, rc);

            break;

        case 0x3C:

            /* CVTQF */

            gen_fcvtqf(rb, rc);

            break;

        case 0x3E:

            /* CVTQG */

            gen_fcvtqg(rb, rc);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x16:

        /* IEEE floating-point */

        switch (fpfn) { /* fn11 & 0x3F */

        case 0x00:

            /* ADDS */

            gen_fadds(ctx, ra, rb, rc, fn11);

            break;

        case 0x01:

            /* SUBS */

            gen_fsubs(ctx, ra, rb, rc, fn11);

            break;

        case 0x02:

            /* MULS */

            gen_fmuls(ctx, ra, rb, rc, fn11);

            break;

        case 0x03:

            /* DIVS */

            gen_fdivs(ctx, ra, rb, rc, fn11);

            break;

        case 0x20:

            /* ADDT */

            gen_faddt(ctx, ra, rb, rc, fn11);

            break;

        case 0x21:

            /* SUBT */

            gen_fsubt(ctx, ra, rb, rc, fn11);

            break;

        case 0x22:

            /* MULT */

            gen_fmult(ctx, ra, rb, rc, fn11);

            break;

        case 0x23:

            /* DIVT */

            gen_fdivt(ctx, ra, rb, rc, fn11);

            break;

        case 0x24:

            /* CMPTUN */

            gen_fcmptun(ctx, ra, rb, rc, fn11);

            break;

        case 0x25:

            /* CMPTEQ */

            gen_fcmpteq(ctx, ra, rb, rc, fn11);

            break;

        case 0x26:

            /* CMPTLT */

            gen_fcmptlt(ctx, ra, rb, rc, fn11);

            break;

        case 0x27:

            /* CMPTLE */

            gen_fcmptle(ctx, ra, rb, rc, fn11);

            break;

        case 0x2C:

            if (fn11 == 0x2AC || fn11 == 0x6AC) {

                /* CVTST */

                gen_fcvtst(ctx, rb, rc, fn11);

            } else {

                /* CVTTS */

                gen_fcvtts(ctx, rb, rc, fn11);

            }

            break;

        case 0x2F:

            /* CVTTQ */

            gen_fcvttq(ctx, rb, rc, fn11);

            break;

        case 0x3C:

            /* CVTQS */

            gen_fcvtqs(ctx, rb, rc, fn11);

            break;

        case 0x3E:

            /* CVTQT */

            gen_fcvtqt(ctx, rb, rc, fn11);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x17:

        switch (fn11) {

        case 0x010:

            /* CVTLQ */

            gen_fcvtlq(rb, rc);

            break;

        case 0x020:

            if (likely(rc != 31)) {

                if (ra == rb) {

                    /* FMOV */

                    if (ra == 31) {

                        tcg_gen_movi_i64(cpu_fir[rc], 0);

                    } else {

                        tcg_gen_mov_i64(cpu_fir[rc], cpu_fir[ra]);

                    }

                } else {

                    /* CPYS */

                    gen_fcpys(ra, rb, rc);

                }

            }

            break;

        case 0x021:

            /* CPYSN */

            gen_fcpysn(ra, rb, rc);

            break;

        case 0x022:

            /* CPYSE */

            gen_fcpyse(ra, rb, rc);

            break;

        case 0x024:

            /* MT_FPCR */

            if (likely(ra != 31)) {

                gen_helper_store_fpcr(cpu_env, cpu_fir[ra]);

            } else {

                TCGv tmp = tcg_const_i64(0);

                gen_helper_store_fpcr(cpu_env, tmp);

                tcg_temp_free(tmp);

            }

            break;

        case 0x025:

            /* MF_FPCR */

            if (likely(ra != 31)) {

                gen_helper_load_fpcr(cpu_fir[ra], cpu_env);

            }

            break;

        case 0x02A:

            /* FCMOVEQ */

            gen_fcmov(TCG_COND_EQ, ra, rb, rc);

            break;

        case 0x02B:

            /* FCMOVNE */

            gen_fcmov(TCG_COND_NE, ra, rb, rc);

            break;

        case 0x02C:

            /* FCMOVLT */

            gen_fcmov(TCG_COND_LT, ra, rb, rc);

            break;

        case 0x02D:

            /* FCMOVGE */

            gen_fcmov(TCG_COND_GE, ra, rb, rc);

            break;

        case 0x02E:

            /* FCMOVLE */

            gen_fcmov(TCG_COND_LE, ra, rb, rc);

            break;

        case 0x02F:

            /* FCMOVGT */

            gen_fcmov(TCG_COND_GT, ra, rb, rc);

            break;

        case 0x030:

            /* CVTQL */

            gen_fcvtql(rb, rc);

            break;

        case 0x130:

            /* CVTQL/V */

        case 0x530:

            /* CVTQL/SV */

            /* ??? I'm pretty sure there's nothing that /sv needs to do that

               /v doesn't do.  The only thing I can think is that /sv is a

               valid instruction merely for completeness in the ISA.  */

            gen_fcvtql_v(ctx, rb, rc);

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x18:

        switch ((uint16_t)disp16) {

        case 0x0000:

            /* TRAPB */

            /* No-op.  */

            break;

        case 0x0400:

            /* EXCB */

            /* No-op.  */

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

            if (ra != 31) {

                if (use_icount) {

                    gen_io_start();

                    gen_helper_load_pcc(cpu_ir[ra], cpu_env);

                    gen_io_end();

                    ret = EXIT_PC_STALE;

                } else {

                    gen_helper_load_pcc(cpu_ir[ra], cpu_env);

                }

            }

            break;

        case 0xE000:

            /* RC */

            gen_rx(ra, 0);

            break;

        case 0xE800:

            /* ECB */

            break;

        case 0xF000:

            /* RS */

            gen_rx(ra, 1);

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

#ifndef CONFIG_USER_ONLY

        if (ctx->tb->flags & TB_FLAGS_PAL_MODE) {

            return gen_mfpr(ra, insn & 0xffff);

        }

#endif

        goto invalid_opc;

    case 0x1A:

        /* JMP, JSR, RET, JSR_COROUTINE.  These only differ by the branch

           prediction stack action, which of course we don't implement.  */

        if (rb != 31) {

            tcg_gen_andi_i64(cpu_pc, cpu_ir[rb], ~3);

        } else {

            tcg_gen_movi_i64(cpu_pc, 0);

        }

        if (ra != 31) {

            tcg_gen_movi_i64(cpu_ir[ra], ctx->pc);

        }

        ret = EXIT_PC_UPDATED;

        break;

    case 0x1B:

        /* HW_LD (PALcode) */

#ifndef CONFIG_USER_ONLY

        if (ctx->tb->flags & TB_FLAGS_PAL_MODE) {

            TCGv addr;



            if (ra == 31) {

                break;

            }



            addr = tcg_temp_new();

            if (rb != 31) {

                tcg_gen_addi_i64(addr, cpu_ir[rb], disp12);

            } else {

                tcg_gen_movi_i64(addr, disp12);

            }

            switch ((insn >> 12) & 0xF) {

            case 0x0:

                /* Longword physical access (hw_ldl/p) */

                gen_helper_ldl_phys(cpu_ir[ra], cpu_env, addr);

                break;

            case 0x1:

                /* Quadword physical access (hw_ldq/p) */

                gen_helper_ldq_phys(cpu_ir[ra], cpu_env, addr);

                break;

            case 0x2:

                /* Longword physical access with lock (hw_ldl_l/p) */

                gen_helper_ldl_l_phys(cpu_ir[ra], cpu_env, addr);

                break;

            case 0x3:

                /* Quadword physical access with lock (hw_ldq_l/p) */

                gen_helper_ldq_l_phys(cpu_ir[ra], cpu_env, addr);

                break;

            case 0x4:

                /* Longword virtual PTE fetch (hw_ldl/v) */

                goto invalid_opc;

            case 0x5:

                /* Quadword virtual PTE fetch (hw_ldq/v) */

                goto invalid_opc;

                break;

            case 0x6:

                /* Incpu_ir[ra]id */

                goto invalid_opc;

            case 0x7:

                /* Incpu_ir[ra]id */

                goto invalid_opc;

            case 0x8:

                /* Longword virtual access (hw_ldl) */

                goto invalid_opc;

            case 0x9:

                /* Quadword virtual access (hw_ldq) */

                goto invalid_opc;

            case 0xA:

                /* Longword virtual access with protection check (hw_ldl/w) */

                tcg_gen_qemu_ld_i64(cpu_ir[ra], addr, MMU_KERNEL_IDX, MO_LESL);

                break;

            case 0xB:

                /* Quadword virtual access with protection check (hw_ldq/w) */

                tcg_gen_qemu_ld_i64(cpu_ir[ra], addr, MMU_KERNEL_IDX, MO_LEQ);

                break;

            case 0xC:

                /* Longword virtual access with alt access mode (hw_ldl/a)*/

                goto invalid_opc;

            case 0xD:

                /* Quadword virtual access with alt access mode (hw_ldq/a) */

                goto invalid_opc;

            case 0xE:

                /* Longword virtual access with alternate access mode and

                   protection checks (hw_ldl/wa) */

                tcg_gen_qemu_ld_i64(cpu_ir[ra], addr, MMU_USER_IDX, MO_LESL);

                break;

            case 0xF:

                /* Quadword virtual access with alternate access mode and

                   protection checks (hw_ldq/wa) */

                tcg_gen_qemu_ld_i64(cpu_ir[ra], addr, MMU_USER_IDX, MO_LEQ);

                break;

            }

            tcg_temp_free(addr);

            break;

        }

#endif

        goto invalid_opc;

    case 0x1C:

        switch (fn7) {

        case 0x00:

            /* SEXTB */

            if ((ctx->tb->flags & TB_FLAGS_AMASK_BWX) == 0) {

                goto invalid_opc;

            }

            if (likely(rc != 31)) {

                if (islit) {

                    tcg_gen_movi_i64(cpu_ir[rc], (int64_t)((int8_t)lit));

                } else {

                    tcg_gen_ext8s_i64(cpu_ir[rc], cpu_ir[rb]);

                }

            }

            break;

        case 0x01:

            /* SEXTW */

            if (ctx->tb->flags & TB_FLAGS_AMASK_BWX) {

                if (likely(rc != 31)) {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], (int64_t)((int16_t)lit));

                    } else {

                        tcg_gen_ext16s_i64(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

                break;

            }

            goto invalid_opc;

        case 0x30:

            /* CTPOP */

            if (ctx->tb->flags & TB_FLAGS_AMASK_CIX) {

                if (likely(rc != 31)) {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], ctpop64(lit));

                    } else {

                        gen_helper_ctpop(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

                break;

            }

            goto invalid_opc;

        case 0x31:

            /* PERR */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                gen_perr(ra, rb, rc, islit, lit);

                break;

            }

            goto invalid_opc;

        case 0x32:

            /* CTLZ */

            if (ctx->tb->flags & TB_FLAGS_AMASK_CIX) {

                if (likely(rc != 31)) {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], clz64(lit));

                    } else {

                        gen_helper_ctlz(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

                break;

            }

            goto invalid_opc;

        case 0x33:

            /* CTTZ */

            if (ctx->tb->flags & TB_FLAGS_AMASK_CIX) {

                if (likely(rc != 31)) {

                    if (islit) {

                        tcg_gen_movi_i64(cpu_ir[rc], ctz64(lit));

                    } else {

                        gen_helper_cttz(cpu_ir[rc], cpu_ir[rb]);

                    }

                }

                break;

            }

            goto invalid_opc;

        case 0x34:

            /* UNPKBW */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                if (real_islit || ra != 31) {

                    goto invalid_opc;

                }

                gen_unpkbw(rb, rc);

                break;

            }

            goto invalid_opc;

        case 0x35:

            /* UNPKBL */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                if (real_islit || ra != 31) {

                    goto invalid_opc;

                }

                gen_unpkbl(rb, rc);

                break;

            }

            goto invalid_opc;

        case 0x36:

            /* PKWB */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                if (real_islit || ra != 31) {

                    goto invalid_opc;

                }

                gen_pkwb(rb, rc);

                break;

            }

            goto invalid_opc;

        case 0x37:

            /* PKLB */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                if (real_islit || ra != 31) {

                    goto invalid_opc;

                }

                gen_pklb(rb, rc);

                break;

            }

            goto invalid_opc;

        case 0x38:

            /* MINSB8 */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                gen_minsb8(ra, rb, rc, islit, lit);

                break;

            }

            goto invalid_opc;

        case 0x39:

            /* MINSW4 */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                gen_minsw4(ra, rb, rc, islit, lit);

                break;

            }

            goto invalid_opc;

        case 0x3A:

            /* MINUB8 */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                gen_minub8(ra, rb, rc, islit, lit);

                break;

            }

            goto invalid_opc;

        case 0x3B:

            /* MINUW4 */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                gen_minuw4(ra, rb, rc, islit, lit);

                break;

            }

            goto invalid_opc;

        case 0x3C:

            /* MAXUB8 */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                gen_maxub8(ra, rb, rc, islit, lit);

                break;

            }

            goto invalid_opc;

        case 0x3D:

            /* MAXUW4 */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                gen_maxuw4(ra, rb, rc, islit, lit);

                break;

            }

            goto invalid_opc;

        case 0x3E:

            /* MAXSB8 */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                gen_maxsb8(ra, rb, rc, islit, lit);

                break;

            }

            goto invalid_opc;

        case 0x3F:

            /* MAXSW4 */

            if (ctx->tb->flags & TB_FLAGS_AMASK_MVI) {

                gen_maxsw4(ra, rb, rc, islit, lit);

                break;

            }

            goto invalid_opc;

        case 0x70:

            /* FTOIT */

            if ((ctx->tb->flags & TB_FLAGS_AMASK_FIX) == 0) {

                goto invalid_opc;

            }

            if (likely(rc != 31)) {

                if (ra != 31) {

                    tcg_gen_mov_i64(cpu_ir[rc], cpu_fir[ra]);

                } else {

                    tcg_gen_movi_i64(cpu_ir[rc], 0);

                }

            }

            break;

        case 0x78:

            /* FTOIS */

            if ((ctx->tb->flags & TB_FLAGS_AMASK_FIX) == 0) {

                goto invalid_opc;

            }

            if (rc != 31) {

                TCGv_i32 tmp1 = tcg_temp_new_i32();

                if (ra != 31) {

                    gen_helper_s_to_memory(tmp1, cpu_fir[ra]);

                } else {

                    TCGv tmp2 = tcg_const_i64(0);

                    gen_helper_s_to_memory(tmp1, tmp2);

                    tcg_temp_free(tmp2);

                }

                tcg_gen_ext_i32_i64(cpu_ir[rc], tmp1);

                tcg_temp_free_i32(tmp1);

            }

            break;

        default:

            goto invalid_opc;

        }

        break;

    case 0x1D:

        /* HW_MTPR (PALcode) */

#ifndef CONFIG_USER_ONLY

        if (ctx->tb->flags & TB_FLAGS_PAL_MODE) {

            return gen_mtpr(ctx, rb, insn & 0xffff);

        }

#endif

        goto invalid_opc;

    case 0x1E:

        /* HW_RET (PALcode) */

#ifndef CONFIG_USER_ONLY

        if (ctx->tb->flags & TB_FLAGS_PAL_MODE) {

            if (rb == 31) {

                /* Pre-EV6 CPUs interpreted this as HW_REI, loading the return

                   address from EXC_ADDR.  This turns out to be useful for our

                   emulation PALcode, so continue to accept it.  */

                TCGv tmp = tcg_temp_new();

                tcg_gen_ld_i64(tmp, cpu_env, offsetof(CPUAlphaState, exc_addr));

                gen_helper_hw_ret(cpu_env, tmp);

                tcg_temp_free(tmp);

            } else {

                gen_helper_hw_ret(cpu_env, cpu_ir[rb]);

            }

            ret = EXIT_PC_UPDATED;

            break;

        }

#endif

        goto invalid_opc;

    case 0x1F:

        /* HW_ST (PALcode) */

#ifndef CONFIG_USER_ONLY

        if (ctx->tb->flags & TB_FLAGS_PAL_MODE) {

            TCGv addr, val;

            addr = tcg_temp_new();

            if (rb != 31) {

                tcg_gen_addi_i64(addr, cpu_ir[rb], disp12);

            } else {

                tcg_gen_movi_i64(addr, disp12);

            }

            if (ra != 31) {

                val = cpu_ir[ra];

            } else {

                val = tcg_temp_new();

                tcg_gen_movi_i64(val, 0);

            }

            switch ((insn >> 12) & 0xF) {

            case 0x0:

                /* Longword physical access */

                gen_helper_stl_phys(cpu_env, addr, val);

                break;

            case 0x1:

                /* Quadword physical access */

                gen_helper_stq_phys(cpu_env, addr, val);

                break;

            case 0x2:

                /* Longword physical access with lock */

                gen_helper_stl_c_phys(val, cpu_env, addr, val);

                break;

            case 0x3:

                /* Quadword physical access with lock */

                gen_helper_stq_c_phys(val, cpu_env, addr, val);

                break;

            case 0x4:

                /* Longword virtual access */

                goto invalid_opc;

            case 0x5:

                /* Quadword virtual access */

                goto invalid_opc;

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

                goto invalid_opc;

            case 0xD:

                /* Quadword virtual access with alternate access mode */

                goto invalid_opc;

            case 0xE:

                /* Invalid */

                goto invalid_opc;

            case 0xF:

                /* Invalid */

                goto invalid_opc;

            }

            if (ra == 31) {

                tcg_temp_free(val);

            }

            tcg_temp_free(addr);

            break;

        }

#endif

        goto invalid_opc;

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

        ret = gen_store_conditional(ctx, ra, rb, disp16, 0);

        break;

    case 0x2F:

        /* STQ_C */

        ret = gen_store_conditional(ctx, ra, rb, disp16, 1);

        break;

    case 0x30:

        /* BR */

        ret = gen_bdirect(ctx, ra, disp21);

        break;

    case 0x31: /* FBEQ */

        ret = gen_fbcond(ctx, TCG_COND_EQ, ra, disp21);

        break;

    case 0x32: /* FBLT */

        ret = gen_fbcond(ctx, TCG_COND_LT, ra, disp21);

        break;

    case 0x33: /* FBLE */

        ret = gen_fbcond(ctx, TCG_COND_LE, ra, disp21);

        break;

    case 0x34:

        /* BSR */

        ret = gen_bdirect(ctx, ra, disp21);

        break;

    case 0x35: /* FBNE */

        ret = gen_fbcond(ctx, TCG_COND_NE, ra, disp21);

        break;

    case 0x36: /* FBGE */

        ret = gen_fbcond(ctx, TCG_COND_GE, ra, disp21);

        break;

    case 0x37: /* FBGT */

        ret = gen_fbcond(ctx, TCG_COND_GT, ra, disp21);

        break;

    case 0x38:

        /* BLBC */

        ret = gen_bcond(ctx, TCG_COND_EQ, ra, disp21, 1);

        break;

    case 0x39:

        /* BEQ */

        ret = gen_bcond(ctx, TCG_COND_EQ, ra, disp21, 0);

        break;

    case 0x3A:

        /* BLT */

        ret = gen_bcond(ctx, TCG_COND_LT, ra, disp21, 0);

        break;

    case 0x3B:

        /* BLE */

        ret = gen_bcond(ctx, TCG_COND_LE, ra, disp21, 0);

        break;

    case 0x3C:

        /* BLBS */

        ret = gen_bcond(ctx, TCG_COND_NE, ra, disp21, 1);

        break;

    case 0x3D:

        /* BNE */

        ret = gen_bcond(ctx, TCG_COND_NE, ra, disp21, 0);

        break;

    case 0x3E:

        /* BGE */

        ret = gen_bcond(ctx, TCG_COND_GE, ra, disp21, 0);

        break;

    case 0x3F:

        /* BGT */

        ret = gen_bcond(ctx, TCG_COND_GT, ra, disp21, 0);

        break;

    invalid_opc:

        ret = gen_invalid(ctx);

        break;

    }



    return ret;

}
