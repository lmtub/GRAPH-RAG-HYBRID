static int create_ppc_opcodes (CPUPPCState *env, ppc_def_t *def)

{

    opcode_t *opc, *start, *end;



    fill_new_table(env->opcodes, 0x40);

#if defined(PPC_DUMP_CPU)

    printf("* PowerPC instructions for PVR %08x: %s flags %016" PRIx64

           " %08x\n",

           def->pvr, def->name, def->insns_flags, def->flags);

#endif

    if (&opc_start < &opc_end) {

        start = &opc_start;

        end = &opc_end;

    } else {

        start = &opc_end;

        end = &opc_start;

    }

    for (opc = start + 1; opc != end; opc++) {

        if ((opc->handler.type & def->insns_flags) != 0) {

            if (register_insn(env->opcodes, opc) < 0) {

                printf("*** ERROR initializing PowerPC instruction "

                       "0x%02x 0x%02x 0x%02x\n", opc->opc1, opc->opc2,

                       opc->opc3);

                return -1;

            }

#if defined(PPC_DUMP_CPU)

            if (opc1 != 0x00) {

                if (opc->opc3 == 0xFF) {

                    if (opc->opc2 == 0xFF) {

                        printf(" %02x -- -- (%2d ----) : %s\n",

                               opc->opc1, opc->opc1, opc->oname);

                    } else {

                        printf(" %02x %02x -- (%2d %4d) : %s\n",

                               opc->opc1, opc->opc2, opc->opc1, opc->opc2,

                               opc->oname);

                    }

                } else {

                    printf(" %02x %02x %02x (%2d %4d) : %s\n",

                           opc->opc1, opc->opc2, opc->opc3,

                           opc->opc1, (opc->opc3 << 5) | opc->opc2,

                           opc->oname);

                }

            }

#endif

        }

    }

    fix_opcode_tables(env->opcodes);

    fflush(stdout);

    fflush(stderr);



    return 0;

}
