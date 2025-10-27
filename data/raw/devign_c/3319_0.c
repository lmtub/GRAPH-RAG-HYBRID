void cpu_loop(CPUUniCore32State *env)

{

    CPUState *cs = CPU(uc32_env_get_cpu(env));

    int trapnr;

    unsigned int n, insn;

    target_siginfo_t info;



    for (;;) {

        cpu_exec_start(cs);

        trapnr = uc32_cpu_exec(cs);

        cpu_exec_end(cs);

        switch (trapnr) {

        case UC32_EXCP_PRIV:

            {

                /* system call */

                get_user_u32(insn, env->regs[31] - 4);

                n = insn & 0xffffff;



                if (n >= UC32_SYSCALL_BASE) {

                    /* linux syscall */

                    n -= UC32_SYSCALL_BASE;

                    if (n == UC32_SYSCALL_NR_set_tls) {

                            cpu_set_tls(env, env->regs[0]);

                            env->regs[0] = 0;

                    } else {

                        env->regs[0] = do_syscall(env,

                                                  n,

                                                  env->regs[0],

                                                  env->regs[1],

                                                  env->regs[2],

                                                  env->regs[3],

                                                  env->regs[4],

                                                  env->regs[5],

                                                  0, 0);

                    }

                } else {

                    goto error;

                }

            }

            break;

        case UC32_EXCP_DTRAP:

        case UC32_EXCP_ITRAP:

            info.si_signo = TARGET_SIGSEGV;

            info.si_errno = 0;

            /* XXX: check env->error_code */

            info.si_code = TARGET_SEGV_MAPERR;

            info._sifields._sigfault._addr = env->cp0.c4_faultaddr;

            queue_signal(env, info.si_signo, &info);

            break;

        case EXCP_INTERRUPT:

            /* just indicate that signals should be handled asap */

            break;

        case EXCP_DEBUG:

            {

                int sig;



                sig = gdb_handlesig(cs, TARGET_SIGTRAP);

                if (sig) {

                    info.si_signo = sig;

                    info.si_errno = 0;

                    info.si_code = TARGET_TRAP_BRKPT;

                    queue_signal(env, info.si_signo, &info);

                }

            }

            break;

        default:

            goto error;

        }

        process_pending_signals(env);

    }



error:

    fprintf(stderr, "qemu: unhandled CPU exception 0x%x - aborting\n", trapnr);

    cpu_dump_state(cs, stderr, fprintf, 0);

    abort();

}
