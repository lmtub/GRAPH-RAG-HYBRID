void cpu_loop(CPUOpenRISCState *env)

{

    CPUState *cs = CPU(openrisc_env_get_cpu(env));

    int trapnr, gdbsig;



    for (;;) {

        cpu_exec_start(cs);

        trapnr = cpu_openrisc_exec(cs);

        cpu_exec_end(cs);

        gdbsig = 0;



        switch (trapnr) {

        case EXCP_RESET:

            qemu_log("\nReset request, exit, pc is %#x\n", env->pc);

            exit(EXIT_FAILURE);

            break;

        case EXCP_BUSERR:

            qemu_log("\nBus error, exit, pc is %#x\n", env->pc);

            gdbsig = TARGET_SIGBUS;

            break;

        case EXCP_DPF:

        case EXCP_IPF:

            cpu_dump_state(cs, stderr, fprintf, 0);

            gdbsig = TARGET_SIGSEGV;

            break;

        case EXCP_TICK:

            qemu_log("\nTick time interrupt pc is %#x\n", env->pc);

            break;

        case EXCP_ALIGN:

            qemu_log("\nAlignment pc is %#x\n", env->pc);

            gdbsig = TARGET_SIGBUS;

            break;

        case EXCP_ILLEGAL:

            qemu_log("\nIllegal instructionpc is %#x\n", env->pc);

            gdbsig = TARGET_SIGILL;

            break;

        case EXCP_INT:

            qemu_log("\nExternal interruptpc is %#x\n", env->pc);

            break;

        case EXCP_DTLBMISS:

        case EXCP_ITLBMISS:

            qemu_log("\nTLB miss\n");

            break;

        case EXCP_RANGE:

            qemu_log("\nRange\n");

            gdbsig = TARGET_SIGSEGV;

            break;

        case EXCP_SYSCALL:

            env->pc += 4;   /* 0xc00; */

            env->gpr[11] = do_syscall(env,

                                      env->gpr[11], /* return value       */

                                      env->gpr[3],  /* r3 - r7 are params */

                                      env->gpr[4],

                                      env->gpr[5],

                                      env->gpr[6],

                                      env->gpr[7],

                                      env->gpr[8], 0, 0);

            break;

        case EXCP_FPE:

            qemu_log("\nFloating point error\n");

            break;

        case EXCP_TRAP:

            qemu_log("\nTrap\n");

            gdbsig = TARGET_SIGTRAP;

            break;

        case EXCP_NR:

            qemu_log("\nNR\n");

            break;

        default:

            qemu_log("\nqemu: unhandled CPU exception %#x - aborting\n",

                     trapnr);

            cpu_dump_state(cs, stderr, fprintf, 0);

            gdbsig = TARGET_SIGILL;

            break;

        }

        if (gdbsig) {

            gdb_handlesig(cs, gdbsig);

            if (gdbsig != TARGET_SIGTRAP) {

                exit(EXIT_FAILURE);

            }

        }



        process_pending_signals(env);

    }

}
