void call_pal (CPUState *env, int palcode)

{

    target_ulong ret;



    if (logfile != NULL)

        fprintf(logfile, "%s: palcode %02x\n", __func__, palcode);

    switch (palcode) {

    case 0x83:

        /* CALLSYS */

        if (logfile != NULL)

            fprintf(logfile, "CALLSYS n " TARGET_FMT_ld "\n", env->ir[0]);

        ret = do_syscall(env, env->ir[IR_V0], env->ir[IR_A0], env->ir[IR_A1],

                         env->ir[IR_A2], env->ir[IR_A3], env->ir[IR_A4],

                         env->ir[IR_A5]);

        if (ret >= 0) {

            env->ir[IR_A3] = 0;

            env->ir[IR_V0] = ret;

        } else {

            env->ir[IR_A3] = 1;

            env->ir[IR_V0] = -ret;

        }

        break;

    case 0x9E:

        /* RDUNIQUE */

        env->ir[IR_V0] = env->unique;

        if (logfile != NULL)

            fprintf(logfile, "RDUNIQUE: " TARGET_FMT_lx "\n", env->unique);

        break;

    case 0x9F:

        /* WRUNIQUE */

        env->unique = env->ir[IR_A0];

        if (logfile != NULL)

            fprintf(logfile, "WRUNIQUE: " TARGET_FMT_lx "\n", env->unique);

        break;

    default:

        if (logfile != NULL)

            fprintf(logfile, "%s: unhandled palcode %02x\n",

                    __func__, palcode);

        exit(1);

    }

}
