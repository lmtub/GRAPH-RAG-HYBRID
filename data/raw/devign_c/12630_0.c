void cpu_dump_statistics (CPUState *env, FILE*f,

                          int (*cpu_fprintf)(FILE *f, const char *fmt, ...),

                          int flags)

{

#if defined(DO_PPC_STATISTICS)

    opc_handler_t **t1, **t2, **t3, *handler;

    int op1, op2, op3;



    t1 = env->opcodes;

    for (op1 = 0; op1 < 64; op1++) {

        handler = t1[op1];

        if (is_indirect_opcode(handler)) {

            t2 = ind_table(handler);

            for (op2 = 0; op2 < 32; op2++) {

                handler = t2[op2];

                if (is_indirect_opcode(handler)) {

                    t3 = ind_table(handler);

                    for (op3 = 0; op3 < 32; op3++) {

                        handler = t3[op3];

                        if (handler->count == 0)

                            continue;

                        cpu_fprintf(f, "%02x %02x %02x (%02x %04d) %16s: "

                                    "%016" PRIx64 " %" PRId64 "\n",

                                    op1, op2, op3, op1, (op3 << 5) | op2,

                                    handler->oname,

                                    handler->count, handler->count);

                    }

                } else {

                    if (handler->count == 0)

                        continue;

                    cpu_fprintf(f, "%02x %02x    (%02x %04d) %16s: "

                                "%016" PRIx64 " %" PRId64 "\n",

                                op1, op2, op1, op2, handler->oname,

                                handler->count, handler->count);

                }

            }

        } else {

            if (handler->count == 0)

                continue;

            cpu_fprintf(f, "%02x       (%02x     ) %16s: %016" PRIx64

                        " %" PRId64 "\n",

                        op1, op1, handler->oname,

                        handler->count, handler->count);

        }

    }

#endif

}
