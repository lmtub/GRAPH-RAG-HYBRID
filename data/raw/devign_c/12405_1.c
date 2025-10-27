void cpu_loop (CPUSPARCState *env)

{

    int trapnr, ret;

    target_siginfo_t info;



    while (1) {

        trapnr = cpu_sparc_exec (env);



        switch (trapnr) {

#ifndef TARGET_SPARC64

        case 0x88:

        case 0x90:

#else

        case 0x110:

        case 0x16d:

#endif

            ret = do_syscall (env, env->gregs[1],

                              env->regwptr[0], env->regwptr[1],

                              env->regwptr[2], env->regwptr[3],

                              env->regwptr[4], env->regwptr[5]);

            if ((unsigned int)ret >= (unsigned int)(-515)) {

#if defined(TARGET_SPARC64) && !defined(TARGET_ABI32)

                env->xcc |= PSR_CARRY;

#else

                env->psr |= PSR_CARRY;

#endif

                ret = -ret;

            } else {

#if defined(TARGET_SPARC64) && !defined(TARGET_ABI32)

                env->xcc &= ~PSR_CARRY;

#else

                env->psr &= ~PSR_CARRY;

#endif

            }

            env->regwptr[0] = ret;

            /* next instruction */

            env->pc = env->npc;

            env->npc = env->npc + 4;

            break;

        case 0x83: /* flush windows */

#ifdef TARGET_ABI32

        case 0x103:

#endif

            flush_windows(env);

            /* next instruction */

            env->pc = env->npc;

            env->npc = env->npc + 4;

            break;

#ifndef TARGET_SPARC64

        case TT_WIN_OVF: /* window overflow */

            save_window(env);

            break;

        case TT_WIN_UNF: /* window underflow */

            restore_window(env);

            break;

        case TT_TFAULT:

        case TT_DFAULT:

            {

                info.si_signo = SIGSEGV;

                info.si_errno = 0;

                /* XXX: check env->error_code */

                info.si_code = TARGET_SEGV_MAPERR;

                info._sifields._sigfault._addr = env->mmuregs[4];

                queue_signal(env, info.si_signo, &info);

            }

            break;

#else

        case TT_SPILL: /* window overflow */

            save_window(env);

            break;

        case TT_FILL: /* window underflow */

            restore_window(env);

            break;

        case TT_TFAULT:

        case TT_DFAULT:

            {

                info.si_signo = SIGSEGV;

                info.si_errno = 0;

                /* XXX: check env->error_code */

                info.si_code = TARGET_SEGV_MAPERR;

                if (trapnr == TT_DFAULT)

                    info._sifields._sigfault._addr = env->dmmuregs[4];

                else

                    info._sifields._sigfault._addr = env->tsptr->tpc;

                queue_signal(env, info.si_signo, &info);

            }

            break;

#ifndef TARGET_ABI32

        case 0x16e:

            flush_windows(env);

            sparc64_get_context(env);

            break;

        case 0x16f:

            flush_windows(env);

            sparc64_set_context(env);

            break;

#endif

#endif

        case EXCP_INTERRUPT:

            /* just indicate that signals should be handled asap */

            break;

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

                  }

            }

            break;

        default:

            printf ("Unhandled trap: 0x%x\n", trapnr);

            cpu_dump_state(env, stderr, fprintf, 0);

            exit (1);

        }

        process_pending_signals (env);

    }

}
