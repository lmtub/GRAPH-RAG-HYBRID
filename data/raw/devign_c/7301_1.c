int kvm_arch_debug(struct kvm_debug_exit_arch *arch_info)

{

    int handle = 0;

    int n;



    if (arch_info->exception == 1) {

        if (arch_info->dr6 & (1 << 14)) {

            if (cpu_single_env->singlestep_enabled)

                handle = 1;

        } else {

            for (n = 0; n < 4; n++)

                if (arch_info->dr6 & (1 << n))

                    switch ((arch_info->dr7 >> (16 + n*4)) & 0x3) {

                    case 0x0:

                        handle = 1;

                        break;

                    case 0x1:

                        handle = 1;

                        cpu_single_env->watchpoint_hit = &hw_watchpoint;

                        hw_watchpoint.vaddr = hw_breakpoint[n].addr;

                        hw_watchpoint.flags = BP_MEM_WRITE;

                        break;

                    case 0x3:

                        handle = 1;

                        cpu_single_env->watchpoint_hit = &hw_watchpoint;

                        hw_watchpoint.vaddr = hw_breakpoint[n].addr;

                        hw_watchpoint.flags = BP_MEM_ACCESS;

                        break;

                    }

        }

    } else if (kvm_find_sw_breakpoint(cpu_single_env, arch_info->pc))

        handle = 1;



    if (!handle)

        kvm_update_guest_debug(cpu_single_env,

                        (arch_info->exception == 1) ?

                        KVM_GUESTDBG_INJECT_DB : KVM_GUESTDBG_INJECT_BP);



    return handle;

}
