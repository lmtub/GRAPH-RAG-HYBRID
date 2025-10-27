void cpu_dump_state(CPUXtensaState *env, FILE *f, fprintf_function cpu_fprintf,

        int flags)

{

    int i, j;



    cpu_fprintf(f, "PC=%08x\n\n", env->pc);



    for (i = j = 0; i < 256; ++i) {

        if (sregnames[i]) {

            cpu_fprintf(f, "%s=%08x%c", sregnames[i], env->sregs[i],

                    (j++ % 4) == 3 ? '\n' : ' ');

        }

    }



    cpu_fprintf(f, (j % 4) == 0 ? "\n" : "\n\n");



    for (i = j = 0; i < 256; ++i) {

        if (uregnames[i]) {

            cpu_fprintf(f, "%s=%08x%c", uregnames[i], env->uregs[i],

                    (j++ % 4) == 3 ? '\n' : ' ');

        }

    }



    cpu_fprintf(f, (j % 4) == 0 ? "\n" : "\n\n");



    for (i = 0; i < 16; ++i) {

        cpu_fprintf(f, "A%02d=%08x%c", i, env->regs[i],

                (i % 4) == 3 ? '\n' : ' ');

    }



    cpu_fprintf(f, "\n");



    for (i = 0; i < env->config->nareg; ++i) {

        cpu_fprintf(f, "AR%02d=%08x%c", i, env->phys_regs[i],

                (i % 4) == 3 ? '\n' : ' ');

    }



    if (xtensa_option_enabled(env->config, XTENSA_OPTION_FP_COPROCESSOR)) {

        cpu_fprintf(f, "\n");



        for (i = 0; i < 16; ++i) {

            cpu_fprintf(f, "F%02d=%08x (%+10.8e)%c", i,

                    float32_val(env->fregs[i]),

                    *(float *)&env->fregs[i], (i % 2) == 1 ? '\n' : ' ');

        }

    }

}
