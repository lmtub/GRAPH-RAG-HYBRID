static int handle_sw_breakpoint(S390CPU *cpu, struct kvm_run *run)

{

    CPUS390XState *env = &cpu->env;

    unsigned long pc;



    cpu_synchronize_state(CPU(cpu));



    pc = env->psw.addr - 4;

    if (kvm_find_sw_breakpoint(CPU(cpu), pc)) {

        env->psw.addr = pc;

        return EXCP_DEBUG;

    }



    return -ENOENT;

}
