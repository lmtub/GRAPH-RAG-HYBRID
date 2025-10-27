static inline void cpu_handle_interrupt(CPUState *cpu,

                                        TranslationBlock **last_tb)

{

    CPUClass *cc = CPU_GET_CLASS(cpu);

    int interrupt_request = cpu->interrupt_request;



    if (unlikely(interrupt_request)) {

        if (unlikely(cpu->singlestep_enabled & SSTEP_NOIRQ)) {

            /* Mask out external interrupts for this step. */

            interrupt_request &= ~CPU_INTERRUPT_SSTEP_MASK;

        }

        if (interrupt_request & CPU_INTERRUPT_DEBUG) {

            cpu->interrupt_request &= ~CPU_INTERRUPT_DEBUG;

            cpu->exception_index = EXCP_DEBUG;

            cpu_loop_exit(cpu);

        }

        if (replay_mode == REPLAY_MODE_PLAY && !replay_has_interrupt()) {

            /* Do nothing */

        } else if (interrupt_request & CPU_INTERRUPT_HALT) {

            replay_interrupt();

            cpu->interrupt_request &= ~CPU_INTERRUPT_HALT;

            cpu->halted = 1;

            cpu->exception_index = EXCP_HLT;

            cpu_loop_exit(cpu);

        }

#if defined(TARGET_I386)

        else if (interrupt_request & CPU_INTERRUPT_INIT) {

            X86CPU *x86_cpu = X86_CPU(cpu);

            CPUArchState *env = &x86_cpu->env;

            replay_interrupt();

            cpu_svm_check_intercept_param(env, SVM_EXIT_INIT, 0);

            do_cpu_init(x86_cpu);

            cpu->exception_index = EXCP_HALTED;

            cpu_loop_exit(cpu);

        }

#else

        else if (interrupt_request & CPU_INTERRUPT_RESET) {

            replay_interrupt();

            cpu_reset(cpu);

            cpu_loop_exit(cpu);

        }

#endif

        /* The target hook has 3 exit conditions:

           False when the interrupt isn't processed,

           True when it is, and we should restart on a new TB,

           and via longjmp via cpu_loop_exit.  */

        else {

            replay_interrupt();

            if (cc->cpu_exec_interrupt(cpu, interrupt_request)) {

                *last_tb = NULL;

            }

            /* The target hook may have updated the 'cpu->interrupt_request';

             * reload the 'interrupt_request' value */

            interrupt_request = cpu->interrupt_request;

        }

        if (interrupt_request & CPU_INTERRUPT_EXITTB) {

            cpu->interrupt_request &= ~CPU_INTERRUPT_EXITTB;

            /* ensure that no TB jump will be modified as

               the program flow was changed */

            *last_tb = NULL;

        }

    }

    if (unlikely(cpu->exit_request || replay_has_interrupt())) {

        cpu->exit_request = 0;

        cpu->exception_index = EXCP_INTERRUPT;

        cpu_loop_exit(cpu);

    }

}
