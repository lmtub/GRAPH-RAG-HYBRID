void do_store_dcr (void)

{

    if (unlikely(env->dcr_env == NULL)) {

        if (loglevel != 0) {

            fprintf(logfile, "No DCR environment\n");

        }

        do_raise_exception_err(EXCP_PROGRAM, EXCP_INVAL | EXCP_INVAL_INVAL);

    } else if (unlikely(ppc_dcr_write(env->dcr_env, T0, T1) != 0)) {

        if (loglevel != 0) {

            fprintf(logfile, "DCR write error %d %03x\n", (int)T0, (int)T0);

        }

        do_raise_exception_err(EXCP_PROGRAM, EXCP_INVAL | EXCP_PRIV_REG);

    }

}
