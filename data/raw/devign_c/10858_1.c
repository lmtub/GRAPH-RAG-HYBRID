cpu_x86_dump_seg_cache(CPUX86State *env, FILE *f, fprintf_function cpu_fprintf,

                       const char *name, struct SegmentCache *sc)

{

#ifdef TARGET_X86_64

    if (env->hflags & HF_CS64_MASK) {

        cpu_fprintf(f, "%-3s=%04x %016" PRIx64 " %08x %08x", name,

                    sc->selector, sc->base, sc->limit, sc->flags & 0x00ffff00);

    } else

#endif

    {

        cpu_fprintf(f, "%-3s=%04x %08x %08x %08x", name, sc->selector,

                    (uint32_t)sc->base, sc->limit, sc->flags & 0x00ffff00);

    }



    if (!(env->hflags & HF_PE_MASK) || !(sc->flags & DESC_P_MASK))

        goto done;



    cpu_fprintf(f, " DPL=%d ", (sc->flags & DESC_DPL_MASK) >> DESC_DPL_SHIFT);

    if (sc->flags & DESC_S_MASK) {

        if (sc->flags & DESC_CS_MASK) {

            cpu_fprintf(f, (sc->flags & DESC_L_MASK) ? "CS64" :

                           ((sc->flags & DESC_B_MASK) ? "CS32" : "CS16"));

            cpu_fprintf(f, " [%c%c", (sc->flags & DESC_C_MASK) ? 'C' : '-',

                        (sc->flags & DESC_R_MASK) ? 'R' : '-');

        } else {

            cpu_fprintf(f, (sc->flags & DESC_B_MASK) ? "DS  " : "DS16");

            cpu_fprintf(f, " [%c%c", (sc->flags & DESC_E_MASK) ? 'E' : '-',

                        (sc->flags & DESC_W_MASK) ? 'W' : '-');

        }

        cpu_fprintf(f, "%c]", (sc->flags & DESC_A_MASK) ? 'A' : '-');

    } else {

        static const char *sys_type_name[2][16] = {

            { /* 32 bit mode */

                "Reserved", "TSS16-avl", "LDT", "TSS16-busy",

                "CallGate16", "TaskGate", "IntGate16", "TrapGate16",

                "Reserved", "TSS32-avl", "Reserved", "TSS32-busy",

                "CallGate32", "Reserved", "IntGate32", "TrapGate32"

            },

            { /* 64 bit mode */

                "<hiword>", "Reserved", "LDT", "Reserved", "Reserved",

                "Reserved", "Reserved", "Reserved", "Reserved",

                "TSS64-avl", "Reserved", "TSS64-busy", "CallGate64",

                "Reserved", "IntGate64", "TrapGate64"

            }

        };

        cpu_fprintf(f, "%s",

                    sys_type_name[(env->hflags & HF_LMA_MASK) ? 1 : 0]

                                 [(sc->flags & DESC_TYPE_MASK)

                                  >> DESC_TYPE_SHIFT]);

    }

done:

    cpu_fprintf(f, "\n");

}
