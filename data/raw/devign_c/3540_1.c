void cpu_loop(CPUMIPSState *env)

{

    target_siginfo_t info;

    int trapnr, ret;

    unsigned int syscall_num;



    for(;;) {

        trapnr = cpu_mips_exec(env);

        switch(trapnr) {

        case EXCP_SYSCALL:

            syscall_num = env->active_tc.gpr[2] - 4000;

            env->active_tc.PC += 4;

            if (syscall_num >= sizeof(mips_syscall_args)) {

                ret = -ENOSYS;

            } else {

                int nb_args;

                abi_ulong sp_reg;

                abi_ulong arg5 = 0, arg6 = 0, arg7 = 0, arg8 = 0;



                nb_args = mips_syscall_args[syscall_num];

                sp_reg = env->active_tc.gpr[29];

                switch (nb_args) {

                /* these arguments are taken from the stack */

                /* FIXME - what to do if get_user() fails? */

                case 8: get_user_ual(arg8, sp_reg + 28);

                case 7: get_user_ual(arg7, sp_reg + 24);

                case 6: get_user_ual(arg6, sp_reg + 20);

                case 5: get_user_ual(arg5, sp_reg + 16);

                default:



                ret = do_syscall(env, env->active_tc.gpr[2],

                                 env->active_tc.gpr[4],

                                 env->active_tc.gpr[5],

                                 env->active_tc.gpr[6],

                                 env->active_tc.gpr[7],

                                 arg5, arg6/*, arg7, arg8*/);







            if ((unsigned int)ret >= (unsigned int)(-1133)) {

                env->active_tc.gpr[7] = 1; /* error flag */

                ret = -ret;

            } else {

                env->active_tc.gpr[7] = 0; /* error flag */


            env->active_tc.gpr[2] = ret;


        case EXCP_TLBL:

        case EXCP_TLBS:

            info.si_signo = TARGET_SIGSEGV;

            info.si_errno = 0;

            /* XXX: check env->error_code */

            info.si_code = TARGET_SEGV_MAPERR;

            info._sifields._sigfault._addr = env->CP0_BadVAddr;

            queue_signal(env, info.si_signo, &info);


        case EXCP_CpU:

        case EXCP_RI:

            info.si_signo = TARGET_SIGILL;

            info.si_errno = 0;

            info.si_code = 0;

            queue_signal(env, info.si_signo, &info);


        case EXCP_INTERRUPT:

            /* just indicate that signals should be handled asap */


        case EXCP_DEBUG:

            {

                int sig;



                sig = gdb_handlesig (env, TARGET_SIGTRAP);

                if (sig)

                  {

                    info.si_signo = sig;

                    info.si_errno = 0;

                    info.si_code = TARGET_TRAP_BRKPT;

                    queue_signal(env, info.si_signo, &info);




        default:

            //        error:

            fprintf(stderr, "qemu: unhandled CPU exception 0x%x - aborting\n",

                    trapnr);

            cpu_dump_state(env, stderr, fprintf, 0);

            abort();


        process_pending_signals(env);

