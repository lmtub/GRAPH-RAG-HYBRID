static void gen_loongson_multimedia(DisasContext *ctx, int rd, int rs, int rt)

{

    const char *opn = "loongson_cp2";

    uint32_t opc, shift_max;

    TCGv_i64 t0, t1;



    opc = MASK_LMI(ctx->opcode);

    switch (opc) {

    case OPC_ADD_CP2:

    case OPC_SUB_CP2:

    case OPC_DADD_CP2:

    case OPC_DSUB_CP2:

        t0 = tcg_temp_local_new_i64();

        t1 = tcg_temp_local_new_i64();

        break;

    default:

        t0 = tcg_temp_new_i64();

        t1 = tcg_temp_new_i64();

        break;

    }



    gen_load_fpr64(ctx, t0, rs);

    gen_load_fpr64(ctx, t1, rt);



#define LMI_HELPER(UP, LO) \

    case OPC_##UP: gen_helper_##LO(t0, t0, t1); opn = #LO; break

#define LMI_HELPER_1(UP, LO) \

    case OPC_##UP: gen_helper_##LO(t0, t0); opn = #LO; break

#define LMI_DIRECT(UP, LO, OP) \

    case OPC_##UP: tcg_gen_##OP##_i64(t0, t0, t1); opn = #LO; break



    switch (opc) {

    LMI_HELPER(PADDSH, paddsh);

    LMI_HELPER(PADDUSH, paddush);

    LMI_HELPER(PADDH, paddh);

    LMI_HELPER(PADDW, paddw);

    LMI_HELPER(PADDSB, paddsb);

    LMI_HELPER(PADDUSB, paddusb);

    LMI_HELPER(PADDB, paddb);



    LMI_HELPER(PSUBSH, psubsh);

    LMI_HELPER(PSUBUSH, psubush);

    LMI_HELPER(PSUBH, psubh);

    LMI_HELPER(PSUBW, psubw);

    LMI_HELPER(PSUBSB, psubsb);

    LMI_HELPER(PSUBUSB, psubusb);

    LMI_HELPER(PSUBB, psubb);



    LMI_HELPER(PSHUFH, pshufh);

    LMI_HELPER(PACKSSWH, packsswh);

    LMI_HELPER(PACKSSHB, packsshb);

    LMI_HELPER(PACKUSHB, packushb);



    LMI_HELPER(PUNPCKLHW, punpcklhw);

    LMI_HELPER(PUNPCKHHW, punpckhhw);

    LMI_HELPER(PUNPCKLBH, punpcklbh);

    LMI_HELPER(PUNPCKHBH, punpckhbh);

    LMI_HELPER(PUNPCKLWD, punpcklwd);

    LMI_HELPER(PUNPCKHWD, punpckhwd);



    LMI_HELPER(PAVGH, pavgh);

    LMI_HELPER(PAVGB, pavgb);

    LMI_HELPER(PMAXSH, pmaxsh);

    LMI_HELPER(PMINSH, pminsh);

    LMI_HELPER(PMAXUB, pmaxub);

    LMI_HELPER(PMINUB, pminub);



    LMI_HELPER(PCMPEQW, pcmpeqw);

    LMI_HELPER(PCMPGTW, pcmpgtw);

    LMI_HELPER(PCMPEQH, pcmpeqh);

    LMI_HELPER(PCMPGTH, pcmpgth);

    LMI_HELPER(PCMPEQB, pcmpeqb);

    LMI_HELPER(PCMPGTB, pcmpgtb);



    LMI_HELPER(PSLLW, psllw);

    LMI_HELPER(PSLLH, psllh);

    LMI_HELPER(PSRLW, psrlw);

    LMI_HELPER(PSRLH, psrlh);

    LMI_HELPER(PSRAW, psraw);

    LMI_HELPER(PSRAH, psrah);



    LMI_HELPER(PMULLH, pmullh);

    LMI_HELPER(PMULHH, pmulhh);

    LMI_HELPER(PMULHUH, pmulhuh);

    LMI_HELPER(PMADDHW, pmaddhw);



    LMI_HELPER(PASUBUB, pasubub);

    LMI_HELPER_1(BIADD, biadd);

    LMI_HELPER_1(PMOVMSKB, pmovmskb);



    LMI_DIRECT(PADDD, paddd, add);

    LMI_DIRECT(PSUBD, psubd, sub);

    LMI_DIRECT(XOR_CP2, xor, xor);

    LMI_DIRECT(NOR_CP2, nor, nor);

    LMI_DIRECT(AND_CP2, and, and);

    LMI_DIRECT(PANDN, pandn, andc);

    LMI_DIRECT(OR, or, or);



    case OPC_PINSRH_0:

        tcg_gen_deposit_i64(t0, t0, t1, 0, 16);

        opn = "pinsrh_0";

        break;

    case OPC_PINSRH_1:

        tcg_gen_deposit_i64(t0, t0, t1, 16, 16);

        opn = "pinsrh_1";

        break;

    case OPC_PINSRH_2:

        tcg_gen_deposit_i64(t0, t0, t1, 32, 16);

        opn = "pinsrh_2";

        break;

    case OPC_PINSRH_3:

        tcg_gen_deposit_i64(t0, t0, t1, 48, 16);

        opn = "pinsrh_3";

        break;



    case OPC_PEXTRH:

        tcg_gen_andi_i64(t1, t1, 3);

        tcg_gen_shli_i64(t1, t1, 4);

        tcg_gen_shr_i64(t0, t0, t1);

        tcg_gen_ext16u_i64(t0, t0);

        opn = "pextrh";

        break;



    case OPC_ADDU_CP2:

        tcg_gen_add_i64(t0, t0, t1);

        tcg_gen_ext32s_i64(t0, t0);

        opn = "addu";

        break;

    case OPC_SUBU_CP2:

        tcg_gen_sub_i64(t0, t0, t1);

        tcg_gen_ext32s_i64(t0, t0);

        opn = "addu";

        break;



    case OPC_SLL_CP2:

        opn = "sll";

        shift_max = 32;

        goto do_shift;

    case OPC_SRL_CP2:

        opn = "srl";

        shift_max = 32;

        goto do_shift;

    case OPC_SRA_CP2:

        opn = "sra";

        shift_max = 32;

        goto do_shift;

    case OPC_DSLL_CP2:

        opn = "dsll";

        shift_max = 64;

        goto do_shift;

    case OPC_DSRL_CP2:

        opn = "dsrl";

        shift_max = 64;

        goto do_shift;

    case OPC_DSRA_CP2:

        opn = "dsra";

        shift_max = 64;

        goto do_shift;

    do_shift:

        /* Make sure shift count isn't TCG undefined behaviour.  */

        tcg_gen_andi_i64(t1, t1, shift_max - 1);



        switch (opc) {

        case OPC_SLL_CP2:

        case OPC_DSLL_CP2:

            tcg_gen_shl_i64(t0, t0, t1);

            break;

        case OPC_SRA_CP2:

        case OPC_DSRA_CP2:

            /* Since SRA is UndefinedResult without sign-extended inputs,

               we can treat SRA and DSRA the same.  */

            tcg_gen_sar_i64(t0, t0, t1);

            break;

        case OPC_SRL_CP2:

            /* We want to shift in zeros for SRL; zero-extend first.  */

            tcg_gen_ext32u_i64(t0, t0);

            /* FALLTHRU */

        case OPC_DSRL_CP2:

            tcg_gen_shr_i64(t0, t0, t1);

            break;

        }



        if (shift_max == 32) {

            tcg_gen_ext32s_i64(t0, t0);

        }



        /* Shifts larger than MAX produce zero.  */

        tcg_gen_setcondi_i64(TCG_COND_LTU, t1, t1, shift_max);

        tcg_gen_neg_i64(t1, t1);

        tcg_gen_and_i64(t0, t0, t1);

        break;



    case OPC_ADD_CP2:

    case OPC_DADD_CP2:

        {

            TCGv_i64 t2 = tcg_temp_new_i64();

            int lab = gen_new_label();



            tcg_gen_mov_i64(t2, t0);

            tcg_gen_add_i64(t0, t1, t2);

            if (opc == OPC_ADD_CP2) {

                tcg_gen_ext32s_i64(t0, t0);

            }

            tcg_gen_xor_i64(t1, t1, t2);

            tcg_gen_xor_i64(t2, t2, t0);

            tcg_gen_andc_i64(t1, t2, t1);

            tcg_temp_free_i64(t2);

            tcg_gen_brcondi_i64(TCG_COND_GE, t1, 0, lab);

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(lab);



            opn = (opc == OPC_ADD_CP2 ? "add" : "dadd");

            break;

        }



    case OPC_SUB_CP2:

    case OPC_DSUB_CP2:

        {

            TCGv_i64 t2 = tcg_temp_new_i64();

            int lab = gen_new_label();



            tcg_gen_mov_i64(t2, t0);

            tcg_gen_sub_i64(t0, t1, t2);

            if (opc == OPC_SUB_CP2) {

                tcg_gen_ext32s_i64(t0, t0);

            }

            tcg_gen_xor_i64(t1, t1, t2);

            tcg_gen_xor_i64(t2, t2, t0);

            tcg_gen_and_i64(t1, t1, t2);

            tcg_temp_free_i64(t2);

            tcg_gen_brcondi_i64(TCG_COND_GE, t1, 0, lab);

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(lab);



            opn = (opc == OPC_SUB_CP2 ? "sub" : "dsub");

            break;

        }



    case OPC_PMULUW:

        tcg_gen_ext32u_i64(t0, t0);

        tcg_gen_ext32u_i64(t1, t1);

        tcg_gen_mul_i64(t0, t0, t1);

        opn = "pmuluw";

        break;



    case OPC_SEQU_CP2:

    case OPC_SEQ_CP2:

    case OPC_SLTU_CP2:

    case OPC_SLT_CP2:

    case OPC_SLEU_CP2:

    case OPC_SLE_CP2:

        /* ??? Document is unclear: Set FCC[CC].  Does that mean the

           FD field is the CC field?  */

    default:

        MIPS_INVAL(opn);

        generate_exception(ctx, EXCP_RI);

        return;

    }



#undef LMI_HELPER

#undef LMI_DIRECT



    gen_store_fpr64(ctx, t0, rd);



    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s, %s, %s", opn,

               fregnames[rd], fregnames[rs], fregnames[rt]);

    tcg_temp_free_i64(t0);

    tcg_temp_free_i64(t1);

}
