static int kvm_handle_debug(PowerPCCPU *cpu, struct kvm_run *run)

{

    CPUState *cs = CPU(cpu);

    CPUPPCState *env = &cpu->env;

    struct kvm_debug_exit_arch *arch_info = &run->debug.arch;

    int handle = 0;



    if (kvm_find_sw_breakpoint(cs, arch_info->address)) {

        handle = 1;

    } else {

        /* QEMU is not able to handle debug exception, so inject

         * program exception to guest;

         * Yes program exception NOT debug exception !!

         * For software breakpoint QEMU uses a privileged instruction;

         * So there cannot be any reason that we are here for guest

         * set debug exception, only possibility is guest executed a

         * privileged / illegal instruction and that's why we are

         * injecting a program interrupt.

         */



        cpu_synchronize_state(cs);

        /* env->nip is PC, so increment this by 4 to use

         * ppc_cpu_do_interrupt(), which set srr0 = env->nip - 4.

         */

        env->nip += 4;

        cs->exception_index = POWERPC_EXCP_PROGRAM;

        env->error_code = POWERPC_EXCP_INVAL;

        ppc_cpu_do_interrupt(cs);

    }



    return handle;

}
