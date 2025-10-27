void alpha_translate_init(void)

{

#define DEF_VAR(V)  { &cpu_##V, #V, offsetof(CPUAlphaState, V) }



    typedef struct { TCGv *var; const char *name; int ofs; } GlobalVar;

    static const GlobalVar vars[] = {

        DEF_VAR(pc),

        DEF_VAR(lock_addr),

        DEF_VAR(lock_st_addr),

        DEF_VAR(lock_value),

        DEF_VAR(unique),

#ifndef CONFIG_USER_ONLY

        DEF_VAR(sysval),

        DEF_VAR(usp),

#endif

    };



#undef DEF_VAR



    /* Use the symbolic register names that match the disassembler.  */

    static const char greg_names[31][4] = {

        "v0", "t0", "t1", "t2", "t3", "t4", "t5", "t6",

        "t7", "s0", "s1", "s2", "s3", "s4", "s5", "fp",

        "a0", "a1", "a2", "a3", "a4", "a5", "t8", "t9",

        "t10", "t11", "ra", "t12", "at", "gp", "sp"

    };

    static const char freg_names[31][4] = {

        "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",

        "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15",

        "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",

        "f24", "f25", "f26", "f27", "f28", "f29", "f30"

    };



    static bool done_init = 0;

    int i;



    if (done_init) {

        return;

    }

    done_init = 1;



    cpu_env = tcg_global_reg_new_ptr(TCG_AREG0, "env");



    for (i = 0; i < 31; i++) {

        cpu_ir[i] = tcg_global_mem_new_i64(TCG_AREG0,

                                           offsetof(CPUAlphaState, ir[i]),

                                           greg_names[i]);

    }



    for (i = 0; i < 31; i++) {

        cpu_fir[i] = tcg_global_mem_new_i64(TCG_AREG0,

                                            offsetof(CPUAlphaState, fir[i]),

                                            freg_names[i]);

    }



    for (i = 0; i < ARRAY_SIZE(vars); ++i) {

        const GlobalVar *v = &vars[i];

        *v->var = tcg_global_mem_new_i64(TCG_AREG0, v->ofs, v->name);

    }

}
