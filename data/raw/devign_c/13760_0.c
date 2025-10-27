int gen_intermediate_code_internal (CPUState *env, TranslationBlock *tb,

                                    int search_pc)

{

    DisasContext ctx, *ctxp = &ctx;

    opc_handler_t **table, *handler;

    uint32_t pc_start;

    uint16_t *gen_opc_end;

    int j, lj = -1;



    pc_start = tb->pc;

    gen_opc_ptr = gen_opc_buf;

    gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;

    gen_opparam_ptr = gen_opparam_buf;

    ctx.nip = pc_start;

    ctx.tb = tb;

    ctx.exception = EXCP_NONE;

#if defined(CONFIG_USER_ONLY)

    ctx.mem_idx = 0;

#else

    ctx.supervisor = 1 - msr_pr;

    ctx.mem_idx = (1 - msr_pr);

#endif

#if defined (DO_SINGLE_STEP)

    /* Single step trace mode */

    msr_se = 1;

#endif

    env->access_type = ACCESS_CODE;

    /* Set env in case of segfault during code fetch */

    while (ctx.exception == EXCP_NONE && gen_opc_ptr < gen_opc_end) {

        if (search_pc) {

            if (loglevel > 0)

                fprintf(logfile, "Search PC...\n");

            j = gen_opc_ptr - gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    gen_opc_instr_start[lj++] = 0;

                gen_opc_pc[lj] = ctx.nip;

                gen_opc_instr_start[lj] = 1;

            }

        }

#if defined PPC_DEBUG_DISAS

        if (loglevel & CPU_LOG_TB_IN_ASM) {

            fprintf(logfile, "----------------\n");

            fprintf(logfile, "nip=%08x super=%d ir=%d\n",

                    ctx.nip, 1 - msr_pr, msr_ir);

        }

#endif

        ctx.opcode = ldl_code((void *)ctx.nip);

#if defined PPC_DEBUG_DISAS

        if (loglevel & CPU_LOG_TB_IN_ASM) {

            fprintf(logfile, "translate opcode %08x (%02x %02x %02x)\n",

                    ctx.opcode, opc1(ctx.opcode), opc2(ctx.opcode),

                    opc3(ctx.opcode));

        }

#endif

        ctx.nip += 4;

        table = ppc_opcodes;

        handler = table[opc1(ctx.opcode)];

        if (is_indirect_opcode(handler)) {

            table = ind_table(handler);

            handler = table[opc2(ctx.opcode)];

            if (is_indirect_opcode(handler)) {

                table = ind_table(handler);

                handler = table[opc3(ctx.opcode)];

            }

        }

        /* Is opcode *REALLY* valid ? */

                if (handler->handler == &gen_invalid) {

            if (loglevel > 0) {

                    fprintf(logfile, "invalid/unsupported opcode: "

                        "%02x - %02x - %02x (%08x) 0x%08x %d\n",

                            opc1(ctx.opcode), opc2(ctx.opcode),

                        opc3(ctx.opcode), ctx.opcode, ctx.nip - 4, msr_ir);

            } else {

                printf("invalid/unsupported opcode: "

                       "%02x - %02x - %02x (%08x) 0x%08x %d\n",

                       opc1(ctx.opcode), opc2(ctx.opcode),

                       opc3(ctx.opcode), ctx.opcode, ctx.nip - 4, msr_ir);

            }

                } else {

            if ((ctx.opcode & handler->inval) != 0) {

                if (loglevel > 0) {

                    fprintf(logfile, "invalid bits: %08x for opcode: "

                            "%02x -%02x - %02x (0x%08x) (0x%08x)\n",

                            ctx.opcode & handler->inval, opc1(ctx.opcode),

                            opc2(ctx.opcode), opc3(ctx.opcode),

                            ctx.opcode, ctx.nip - 4);

                } else {

                    printf("invalid bits: %08x for opcode: "

                           "%02x -%02x - %02x (0x%08x) (0x%08x)\n",

                            ctx.opcode & handler->inval, opc1(ctx.opcode),

                            opc2(ctx.opcode), opc3(ctx.opcode),

                           ctx.opcode, ctx.nip - 4);

            }

                RET_INVAL(ctxp);

                break;

            }

        }

        (*(handler->handler))(&ctx);

        /* Check trace mode exceptions */

        if ((msr_be && ctx.exception == EXCP_BRANCH) ||

            /* Check in single step trace mode

             * we need to stop except if:

             * - rfi, trap or syscall

             * - first instruction of an exception handler

             */

            (msr_se && (ctx.nip < 0x100 ||

                        ctx.nip > 0xF00 ||

                        (ctx.nip & 0xFC) != 0x04) &&

             ctx.exception != EXCP_SYSCALL && ctx.exception != EXCP_RFI &&

             ctx.exception != EXCP_TRAP)) {

            RET_EXCP(ctxp, EXCP_TRACE, 0);

        }

        /* if we reach a page boundary, stop generation */

        if ((ctx.nip & (TARGET_PAGE_SIZE - 1)) == 0) {

            RET_EXCP(ctxp, EXCP_BRANCH, 0);

    }

    }

    if (ctx.exception == EXCP_NONE) {

        gen_op_b((unsigned long)ctx.tb, ctx.nip);

    } else if (ctx.exception != EXCP_BRANCH) {

        gen_op_set_T0(0);

    }

#if 1

    /* TO BE FIXED: T0 hasn't got a proper value, which makes tb_add_jump

     *              do bad business and then qemu crashes !

     */

    gen_op_set_T0(0);

#endif

    /* Generate the return instruction */

    gen_op_exit_tb();

    *gen_opc_ptr = INDEX_op_end;

    if (search_pc) {

        j = gen_opc_ptr - gen_opc_buf;

        lj++;

        while (lj <= j)

            gen_opc_instr_start[lj++] = 0;

        tb->size = 0;

#if 0

        if (loglevel > 0) {

            page_dump(logfile);

        }

#endif

    } else {

        tb->size = ctx.nip - pc_start;

    }

#ifdef DEBUG_DISAS

    if (loglevel & CPU_LOG_TB_CPU) {

        fprintf(logfile, "---------------- excp: %04x\n", ctx.exception);

        cpu_ppc_dump_state(env, logfile, 0);

    }

    if (loglevel & CPU_LOG_TB_IN_ASM) {

        fprintf(logfile, "IN: %s\n", lookup_symbol((void *)pc_start));

	disas(logfile, (void *)pc_start, ctx.nip - pc_start, 0, 0);

        fprintf(logfile, "\n");

    }

    if (loglevel & CPU_LOG_TB_OP) {

        fprintf(logfile, "OP:\n");

        dump_ops(gen_opc_buf, gen_opparam_buf);

        fprintf(logfile, "\n");

    }

#endif

    env->access_type = ACCESS_INT;



    return 0;

}
