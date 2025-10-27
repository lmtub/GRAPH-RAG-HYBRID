static void gen_cp0 (CPUMIPSState *env, DisasContext *ctx, uint32_t opc, int rt, int rd)

{

    const char *opn = "ldst";




    switch (opc) {

    case OPC_MFC0:

        if (rt == 0) {

            /* Treat as NOP. */

            return;

        }

        gen_mfc0(env, ctx, cpu_gpr[rt], rd, ctx->opcode & 0x7);

        opn = "mfc0";

        break;

    case OPC_MTC0:

        {

            TCGv t0 = tcg_temp_new();



            gen_load_gpr(t0, rt);

            gen_mtc0(env, ctx, t0, rd, ctx->opcode & 0x7);

            tcg_temp_free(t0);

        }

        opn = "mtc0";

        break;

#if defined(TARGET_MIPS64)

    case OPC_DMFC0:

        check_insn(env, ctx, ISA_MIPS3);

        if (rt == 0) {

            /* Treat as NOP. */

            return;

        }

        gen_dmfc0(env, ctx, cpu_gpr[rt], rd, ctx->opcode & 0x7);

        opn = "dmfc0";

        break;

    case OPC_DMTC0:

        check_insn(env, ctx, ISA_MIPS3);

        {

            TCGv t0 = tcg_temp_new();



            gen_load_gpr(t0, rt);

            gen_dmtc0(env, ctx, t0, rd, ctx->opcode & 0x7);

            tcg_temp_free(t0);

        }

        opn = "dmtc0";

        break;

#endif

    case OPC_MFTR:

        check_insn(env, ctx, ASE_MT);

        if (rd == 0) {

            /* Treat as NOP. */

            return;

        }

        gen_mftr(env, ctx, rt, rd, (ctx->opcode >> 5) & 1,

                 ctx->opcode & 0x7, (ctx->opcode >> 4) & 1);

        opn = "mftr";

        break;

    case OPC_MTTR:

        check_insn(env, ctx, ASE_MT);

        gen_mttr(env, ctx, rd, rt, (ctx->opcode >> 5) & 1,

                 ctx->opcode & 0x7, (ctx->opcode >> 4) & 1);

        opn = "mttr";

        break;

    case OPC_TLBWI:

        opn = "tlbwi";

        if (!env->tlb->helper_tlbwi)

            goto die;

        gen_helper_tlbwi();

        break;

    case OPC_TLBWR:

        opn = "tlbwr";

        if (!env->tlb->helper_tlbwr)

            goto die;

        gen_helper_tlbwr();

        break;

    case OPC_TLBP:

        opn = "tlbp";

        if (!env->tlb->helper_tlbp)

            goto die;

        gen_helper_tlbp();

        break;

    case OPC_TLBR:

        opn = "tlbr";

        if (!env->tlb->helper_tlbr)

            goto die;

        gen_helper_tlbr();

        break;

    case OPC_ERET:

        opn = "eret";

        check_insn(env, ctx, ISA_MIPS2);

        gen_helper_eret();

        ctx->bstate = BS_EXCP;

        break;

    case OPC_DERET:

        opn = "deret";

        check_insn(env, ctx, ISA_MIPS32);

        if (!(ctx->hflags & MIPS_HFLAG_DM)) {

            MIPS_INVAL(opn);

            generate_exception(ctx, EXCP_RI);

        } else {

            gen_helper_deret();

            ctx->bstate = BS_EXCP;

        }

        break;

    case OPC_WAIT:

        opn = "wait";

        check_insn(env, ctx, ISA_MIPS3 | ISA_MIPS32);

        /* If we get an exception, we want to restart at next instruction */

        ctx->pc += 4;

        save_cpu_state(ctx, 1);

        ctx->pc -= 4;

        gen_helper_wait();

        ctx->bstate = BS_EXCP;

        break;

    default:

 die:

        MIPS_INVAL(opn);

        generate_exception(ctx, EXCP_RI);

        return;

    }

    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s %d", opn, regnames[rt], rd);

}