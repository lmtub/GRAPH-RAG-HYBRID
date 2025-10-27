void kvm_remove_all_breakpoints(CPUState *current_env)

{

    struct kvm_sw_breakpoint *bp, *next;

    KVMState *s = current_env->kvm_state;

    CPUState *env;



    QTAILQ_FOREACH_SAFE(bp, &s->kvm_sw_breakpoints, entry, next) {

        if (kvm_arch_remove_sw_breakpoint(current_env, bp) != 0) {

            /* Try harder to find a CPU that currently sees the breakpoint. */

            for (env = first_cpu; env != NULL; env = env->next_cpu) {

                if (kvm_arch_remove_sw_breakpoint(env, bp) == 0)

                    break;

            }

        }

    }

    kvm_arch_remove_all_hw_breakpoints();



    for (env = first_cpu; env != NULL; env = env->next_cpu)

        kvm_update_guest_debug(env, 0);

}
