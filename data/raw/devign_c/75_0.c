static void gen_cp0 (CPUState *env, DisasContext *ctx, uint32_t opc, int rt, int rd)

{

    const char *opn = "ldst";



    switch (opc) {

    case OPC_MFC0:

        if (rt == 0) {

            /* Treat as NOP */

            return;

        }

        gen_mfc0(ctx, rd, ctx->opcode & 0x7);

        gen_op_store_T0_gpr(rt);

        opn = "mfc0";

        break;

    case OPC_MTC0:

        GEN_LOAD_REG_TN(T0, rt);

        gen_mtc0(ctx, rd, ctx->opcode & 0x7);

        opn = "mtc0";

        break;

#ifdef TARGET_MIPS64

    case OPC_DMFC0:

        if (rt == 0) {

            /* Treat as NOP */

            return;

        }

        gen_dmfc0(ctx, rd, ctx->opcode & 0x7);

        gen_op_store_T0_gpr(rt);

        opn = "dmfc0";

        break;

    case OPC_DMTC0:

        GEN_LOAD_REG_TN(T0, rt);

        gen_dmtc0(ctx, rd, ctx->opcode & 0x7);

        opn = "dmtc0";

        break;

#endif

    case OPC_TLBWI:

        opn = "tlbwi";

        if (!env->do_tlbwi)

            goto die;

        gen_op_tlbwi();

        break;

    case OPC_TLBWR:

        opn = "tlbwr";

        if (!env->do_tlbwr)

            goto die;

        gen_op_tlbwr();

        break;

    case OPC_TLBP:

        opn = "tlbp";

        if (!env->do_tlbp)

            goto die;

        gen_op_tlbp();

        break;

    case OPC_TLBR:

        opn = "tlbr";

        if (!env->do_tlbr)

            goto die;

        gen_op_tlbr();

        break;

    case OPC_ERET:

        opn = "eret";

        save_cpu_state(ctx, 0);

        gen_op_eret();

        ctx->bstate = BS_EXCP;

        break;

    case OPC_DERET:

        opn = "deret";

        if (!(ctx->hflags & MIPS_HFLAG_DM)) {

            MIPS_INVAL(opn);

            generate_exception(ctx, EXCP_RI);

        } else {

            save_cpu_state(ctx, 0);

            gen_op_deret();

            ctx->bstate = BS_EXCP;

        }

        break;

    case OPC_WAIT:

        opn = "wait";

        /* If we get an exception, we want to restart at next instruction */

        ctx->pc += 4;

        save_cpu_state(ctx, 1);

        ctx->pc -= 4;

        gen_op_wait();

        ctx->bstate = BS_EXCP;

        break;

    default:

 die:

        MIPS_INVAL(opn);

        generate_exception(ctx, EXCP_RI);

        return;

    }

    MIPS_DEBUG("%s %s %d", opn, regnames[rt], rd);

}
