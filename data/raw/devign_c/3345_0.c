static void gen_compute_branch (DisasContext *ctx, uint32_t opc,

                                int insn_bytes,

                                int rs, int rt, int32_t offset)

{

    target_ulong btgt = -1;

    int blink = 0;

    int bcond_compute = 0;

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();



    if (ctx->hflags & MIPS_HFLAG_BMASK) {

#ifdef MIPS_DEBUG_DISAS

        LOG_DISAS("Branch in delay slot at PC 0x" TARGET_FMT_lx "\n", ctx->pc);

#endif

        generate_exception(ctx, EXCP_RI);

        goto out;

    }



    /* Load needed operands */

    switch (opc) {

    case OPC_BEQ:

    case OPC_BEQL:

    case OPC_BNE:

    case OPC_BNEL:

        /* Compare two registers */

        if (rs != rt) {

            gen_load_gpr(t0, rs);

            gen_load_gpr(t1, rt);

            bcond_compute = 1;

        }

        btgt = ctx->pc + insn_bytes + offset;

        break;

    case OPC_BGEZ:

    case OPC_BGEZAL:

    case OPC_BGEZALL:

    case OPC_BGEZL:

    case OPC_BGTZ:

    case OPC_BGTZL:

    case OPC_BLEZ:

    case OPC_BLEZL:

    case OPC_BLTZ:

    case OPC_BLTZAL:

    case OPC_BLTZALL:

    case OPC_BLTZL:

        /* Compare to zero */

        if (rs != 0) {

            gen_load_gpr(t0, rs);

            bcond_compute = 1;

        }

        btgt = ctx->pc + insn_bytes + offset;

        break;

    case OPC_J:

    case OPC_JAL:

    case OPC_JALX:

        /* Jump to immediate */

        btgt = ((ctx->pc + insn_bytes) & (int32_t)0xF0000000) | (uint32_t)offset;

        break;

    case OPC_JR:

    case OPC_JALR:

    case OPC_JALRC:

        /* Jump to register */

        if (offset != 0 && offset != 16) {

            /* Hint = 0 is JR/JALR, hint 16 is JR.HB/JALR.HB, the

               others are reserved. */

            MIPS_INVAL("jump hint");

            generate_exception(ctx, EXCP_RI);

            goto out;

        }

        gen_load_gpr(btarget, rs);

        break;

    default:

        MIPS_INVAL("branch/jump");

        generate_exception(ctx, EXCP_RI);

        goto out;

    }

    if (bcond_compute == 0) {

        /* No condition to be computed */

        switch (opc) {

        case OPC_BEQ:     /* rx == rx        */

        case OPC_BEQL:    /* rx == rx likely */

        case OPC_BGEZ:    /* 0 >= 0          */

        case OPC_BGEZL:   /* 0 >= 0 likely   */

        case OPC_BLEZ:    /* 0 <= 0          */

        case OPC_BLEZL:   /* 0 <= 0 likely   */

            /* Always take */

            ctx->hflags |= MIPS_HFLAG_B;

            MIPS_DEBUG("balways");

            break;

        case OPC_BGEZAL:  /* 0 >= 0          */

        case OPC_BGEZALL: /* 0 >= 0 likely   */

            /* Always take and link */

            blink = 31;

            ctx->hflags |= MIPS_HFLAG_B;

            MIPS_DEBUG("balways and link");

            break;

        case OPC_BNE:     /* rx != rx        */

        case OPC_BGTZ:    /* 0 > 0           */

        case OPC_BLTZ:    /* 0 < 0           */

            /* Treat as NOP. */

            MIPS_DEBUG("bnever (NOP)");

            goto out;

        case OPC_BLTZAL:  /* 0 < 0           */

            tcg_gen_movi_tl(cpu_gpr[31], ctx->pc + 8);

            MIPS_DEBUG("bnever and link");

            goto out;

        case OPC_BLTZALL: /* 0 < 0 likely */

            tcg_gen_movi_tl(cpu_gpr[31], ctx->pc + 8);

            /* Skip the instruction in the delay slot */

            MIPS_DEBUG("bnever, link and skip");

            ctx->pc += 4;

            goto out;

        case OPC_BNEL:    /* rx != rx likely */

        case OPC_BGTZL:   /* 0 > 0 likely */

        case OPC_BLTZL:   /* 0 < 0 likely */

            /* Skip the instruction in the delay slot */

            MIPS_DEBUG("bnever and skip");

            ctx->pc += 4;

            goto out;

        case OPC_J:

            ctx->hflags |= MIPS_HFLAG_B;

            MIPS_DEBUG("j " TARGET_FMT_lx, btgt);

            break;

        case OPC_JALX:

            ctx->hflags |= MIPS_HFLAG_BX;

            /* Fallthrough */

        case OPC_JAL:

            blink = 31;

            ctx->hflags |= MIPS_HFLAG_B;

            ctx->hflags |= (ctx->hflags & MIPS_HFLAG_M16

                            ? MIPS_HFLAG_BDS16

                            : MIPS_HFLAG_BDS32);

            MIPS_DEBUG("jal " TARGET_FMT_lx, btgt);

            break;

        case OPC_JR:

            ctx->hflags |= MIPS_HFLAG_BR;

            if (ctx->hflags & MIPS_HFLAG_M16)

                ctx->hflags |= MIPS_HFLAG_BDS16;

            MIPS_DEBUG("jr %s", regnames[rs]);

            break;

        case OPC_JALR:

        case OPC_JALRC:

            blink = rt;

            ctx->hflags |= MIPS_HFLAG_BR;

            if (ctx->hflags & MIPS_HFLAG_M16)

                ctx->hflags |= MIPS_HFLAG_BDS16;

            MIPS_DEBUG("jalr %s, %s", regnames[rt], regnames[rs]);

            break;

        default:

            MIPS_INVAL("branch/jump");

            generate_exception(ctx, EXCP_RI);

            goto out;

        }

    } else {

        switch (opc) {

        case OPC_BEQ:

            tcg_gen_setcond_tl(TCG_COND_EQ, bcond, t0, t1);

            MIPS_DEBUG("beq %s, %s, " TARGET_FMT_lx,

                       regnames[rs], regnames[rt], btgt);

            goto not_likely;

        case OPC_BEQL:

            tcg_gen_setcond_tl(TCG_COND_EQ, bcond, t0, t1);

            MIPS_DEBUG("beql %s, %s, " TARGET_FMT_lx,

                       regnames[rs], regnames[rt], btgt);

            goto likely;

        case OPC_BNE:

            tcg_gen_setcond_tl(TCG_COND_NE, bcond, t0, t1);

            MIPS_DEBUG("bne %s, %s, " TARGET_FMT_lx,

                       regnames[rs], regnames[rt], btgt);

            goto not_likely;

        case OPC_BNEL:

            tcg_gen_setcond_tl(TCG_COND_NE, bcond, t0, t1);

            MIPS_DEBUG("bnel %s, %s, " TARGET_FMT_lx,

                       regnames[rs], regnames[rt], btgt);

            goto likely;

        case OPC_BGEZ:

            tcg_gen_setcondi_tl(TCG_COND_GE, bcond, t0, 0);

            MIPS_DEBUG("bgez %s, " TARGET_FMT_lx, regnames[rs], btgt);

            goto not_likely;

        case OPC_BGEZL:

            tcg_gen_setcondi_tl(TCG_COND_GE, bcond, t0, 0);

            MIPS_DEBUG("bgezl %s, " TARGET_FMT_lx, regnames[rs], btgt);

            goto likely;

        case OPC_BGEZAL:

            tcg_gen_setcondi_tl(TCG_COND_GE, bcond, t0, 0);

            MIPS_DEBUG("bgezal %s, " TARGET_FMT_lx, regnames[rs], btgt);

            blink = 31;

            goto not_likely;

        case OPC_BGEZALL:

            tcg_gen_setcondi_tl(TCG_COND_GE, bcond, t0, 0);

            blink = 31;

            MIPS_DEBUG("bgezall %s, " TARGET_FMT_lx, regnames[rs], btgt);

            goto likely;

        case OPC_BGTZ:

            tcg_gen_setcondi_tl(TCG_COND_GT, bcond, t0, 0);

            MIPS_DEBUG("bgtz %s, " TARGET_FMT_lx, regnames[rs], btgt);

            goto not_likely;

        case OPC_BGTZL:

            tcg_gen_setcondi_tl(TCG_COND_GT, bcond, t0, 0);

            MIPS_DEBUG("bgtzl %s, " TARGET_FMT_lx, regnames[rs], btgt);

            goto likely;

        case OPC_BLEZ:

            tcg_gen_setcondi_tl(TCG_COND_LE, bcond, t0, 0);

            MIPS_DEBUG("blez %s, " TARGET_FMT_lx, regnames[rs], btgt);

            goto not_likely;

        case OPC_BLEZL:

            tcg_gen_setcondi_tl(TCG_COND_LE, bcond, t0, 0);

            MIPS_DEBUG("blezl %s, " TARGET_FMT_lx, regnames[rs], btgt);

            goto likely;

        case OPC_BLTZ:

            tcg_gen_setcondi_tl(TCG_COND_LT, bcond, t0, 0);

            MIPS_DEBUG("bltz %s, " TARGET_FMT_lx, regnames[rs], btgt);

            goto not_likely;

        case OPC_BLTZL:

            tcg_gen_setcondi_tl(TCG_COND_LT, bcond, t0, 0);

            MIPS_DEBUG("bltzl %s, " TARGET_FMT_lx, regnames[rs], btgt);

            goto likely;

        case OPC_BLTZAL:

            tcg_gen_setcondi_tl(TCG_COND_LT, bcond, t0, 0);

            blink = 31;

            MIPS_DEBUG("bltzal %s, " TARGET_FMT_lx, regnames[rs], btgt);

        not_likely:

            ctx->hflags |= MIPS_HFLAG_BC;

            break;

        case OPC_BLTZALL:

            tcg_gen_setcondi_tl(TCG_COND_LT, bcond, t0, 0);

            blink = 31;

            MIPS_DEBUG("bltzall %s, " TARGET_FMT_lx, regnames[rs], btgt);

        likely:

            ctx->hflags |= MIPS_HFLAG_BL;

            break;

        default:

            MIPS_INVAL("conditional branch/jump");

            generate_exception(ctx, EXCP_RI);

            goto out;

        }

    }

    MIPS_DEBUG("enter ds: link %d cond %02x target " TARGET_FMT_lx,

               blink, ctx->hflags, btgt);



    ctx->btarget = btgt;

    if (blink > 0) {

        int post_delay = insn_bytes;

        int lowbit = !!(ctx->hflags & MIPS_HFLAG_M16);



        if (opc != OPC_JALRC)

            post_delay += ((ctx->hflags & MIPS_HFLAG_BDS16) ? 2 : 4);



        tcg_gen_movi_tl(cpu_gpr[blink], ctx->pc + post_delay + lowbit);

    }



 out:

    if (insn_bytes == 2)

        ctx->hflags |= MIPS_HFLAG_B16;

    tcg_temp_free(t0);

    tcg_temp_free(t1);

}
