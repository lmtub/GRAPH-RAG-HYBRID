void cpu_loop(CPUARMState *env)

{

    int trapnr;

    unsigned int n, insn;

    target_siginfo_t info;

    uint32_t addr;



    for(;;) {

        cpu_exec_start(env);

        trapnr = cpu_arm_exec(env);

        cpu_exec_end(env);

        switch(trapnr) {

        case EXCP_UDEF:

            {

                TaskState *ts = env->opaque;

                uint32_t opcode;

                int rc;



                /* we handle the FPU emulation here, as Linux */

                /* we get the opcode */

                /* FIXME - what to do if get_user() fails? */

                get_user_u32(opcode, env->regs[15]);



                rc = EmulateAll(opcode, &ts->fpa, env);

                if (rc == 0) { /* illegal instruction */

                    info.si_signo = SIGILL;

                    info.si_errno = 0;

                    info.si_code = TARGET_ILL_ILLOPN;

                    info._sifields._sigfault._addr = env->regs[15];

                    queue_signal(env, info.si_signo, &info);

                } else if (rc < 0) { /* FP exception */

                    int arm_fpe=0;



                    /* translate softfloat flags to FPSR flags */

                    if (-rc & float_flag_invalid)

                      arm_fpe |= BIT_IOC;

                    if (-rc & float_flag_divbyzero)

                      arm_fpe |= BIT_DZC;

                    if (-rc & float_flag_overflow)

                      arm_fpe |= BIT_OFC;

                    if (-rc & float_flag_underflow)

                      arm_fpe |= BIT_UFC;

                    if (-rc & float_flag_inexact)

                      arm_fpe |= BIT_IXC;



                    FPSR fpsr = ts->fpa.fpsr;

                    //printf("fpsr 0x%x, arm_fpe 0x%x\n",fpsr,arm_fpe);



                    if (fpsr & (arm_fpe << 16)) { /* exception enabled? */

                      info.si_signo = SIGFPE;

                      info.si_errno = 0;



                      /* ordered by priority, least first */

                      if (arm_fpe & BIT_IXC) info.si_code = TARGET_FPE_FLTRES;

                      if (arm_fpe & BIT_UFC) info.si_code = TARGET_FPE_FLTUND;

                      if (arm_fpe & BIT_OFC) info.si_code = TARGET_FPE_FLTOVF;

                      if (arm_fpe & BIT_DZC) info.si_code = TARGET_FPE_FLTDIV;

                      if (arm_fpe & BIT_IOC) info.si_code = TARGET_FPE_FLTINV;



                      info._sifields._sigfault._addr = env->regs[15];

                      queue_signal(env, info.si_signo, &info);

                    } else {

                      env->regs[15] += 4;

                    }



                    /* accumulate unenabled exceptions */

                    if ((!(fpsr & BIT_IXE)) && (arm_fpe & BIT_IXC))

                      fpsr |= BIT_IXC;

                    if ((!(fpsr & BIT_UFE)) && (arm_fpe & BIT_UFC))

                      fpsr |= BIT_UFC;

                    if ((!(fpsr & BIT_OFE)) && (arm_fpe & BIT_OFC))

                      fpsr |= BIT_OFC;

                    if ((!(fpsr & BIT_DZE)) && (arm_fpe & BIT_DZC))

                      fpsr |= BIT_DZC;

                    if ((!(fpsr & BIT_IOE)) && (arm_fpe & BIT_IOC))

                      fpsr |= BIT_IOC;

                    ts->fpa.fpsr=fpsr;

                } else { /* everything OK */

                    /* increment PC */

                    env->regs[15] += 4;

                }

            }

            break;

        case EXCP_SWI:

        case EXCP_BKPT:

            {

                env->eabi = 1;

                /* system call */

                if (trapnr == EXCP_BKPT) {

                    if (env->thumb) {

                        /* FIXME - what to do if get_user() fails? */

                        get_user_u16(insn, env->regs[15]);

                        n = insn & 0xff;

                        env->regs[15] += 2;

                    } else {

                        /* FIXME - what to do if get_user() fails? */

                        get_user_u32(insn, env->regs[15]);

                        n = (insn & 0xf) | ((insn >> 4) & 0xff0);

                        env->regs[15] += 4;

                    }

                } else {

                    if (env->thumb) {

                        /* FIXME - what to do if get_user() fails? */

                        get_user_u16(insn, env->regs[15] - 2);

                        n = insn & 0xff;

                    } else {

                        /* FIXME - what to do if get_user() fails? */

                        get_user_u32(insn, env->regs[15] - 4);

                        n = insn & 0xffffff;

                    }

                }



                if (n == ARM_NR_cacheflush) {

                    /* nop */

                } else if (n == ARM_NR_semihosting

                           || n == ARM_NR_thumb_semihosting) {

                    env->regs[0] = do_arm_semihosting (env);

                } else if (n == 0 || n >= ARM_SYSCALL_BASE

                           || (env->thumb && n == ARM_THUMB_SYSCALL)) {

                    /* linux syscall */

                    if (env->thumb || n == 0) {

                        n = env->regs[7];

                    } else {

                        n -= ARM_SYSCALL_BASE;

                        env->eabi = 0;

                    }

                    if ( n > ARM_NR_BASE) {

                        switch (n) {

                        case ARM_NR_cacheflush:

                            /* nop */

                            break;

                        case ARM_NR_set_tls:

                            cpu_set_tls(env, env->regs[0]);

                            env->regs[0] = 0;

                            break;

                        default:

                            gemu_log("qemu: Unsupported ARM syscall: 0x%x\n",

                                     n);

                            env->regs[0] = -TARGET_ENOSYS;

                            break;

                        }

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

        case EXCP_INTERRUPT:

            /* just indicate that signals should be handled asap */

            break;

        case EXCP_PREFETCH_ABORT:

            addr = env->cp15.c6_insn;

            goto do_segv;

        case EXCP_DATA_ABORT:

            addr = env->cp15.c6_data;

        do_segv:

            {

                info.si_signo = SIGSEGV;

                info.si_errno = 0;

                /* XXX: check env->error_code */

                info.si_code = TARGET_SEGV_MAPERR;

                info._sifields._sigfault._addr = addr;

                queue_signal(env, info.si_signo, &info);

            }

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

        case EXCP_KERNEL_TRAP:

            if (do_kernel_trap(env))

              goto error;

            break;

        case EXCP_STREX:

            if (do_strex(env)) {

                addr = env->cp15.c6_data;

                goto do_segv;

            }

            break;

        default:

        error:

            fprintf(stderr, "qemu: unhandled CPU exception 0x%x - aborting\n",

                    trapnr);

            cpu_dump_state(env, stderr, fprintf, 0);

            abort();

        }

        process_pending_signals(env);

    }

}
