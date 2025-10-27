void cpu_x86_inject_mce(Monitor *mon, CPUState *cenv, int bank,

                        uint64_t status, uint64_t mcg_status, uint64_t addr,

                        uint64_t misc, int flags)

{

    unsigned bank_num = cenv->mcg_cap & 0xff;

    CPUState *env;

    int flag = 0;



    if (!cenv->mcg_cap) {

        monitor_printf(mon, "MCE injection not supported\n");

        return;

    }

    if (bank >= bank_num) {

        monitor_printf(mon, "Invalid MCE bank number\n");

        return;

    }

    if (!(status & MCI_STATUS_VAL)) {

        monitor_printf(mon, "Invalid MCE status code\n");

        return;

    }

    if ((flags & MCE_INJECT_BROADCAST)

        && !cpu_x86_support_mca_broadcast(cenv)) {

        monitor_printf(mon, "Guest CPU does not support MCA broadcast\n");

        return;

    }



    if (kvm_enabled()) {

        if (flags & MCE_INJECT_BROADCAST) {

            flag |= MCE_BROADCAST;

        }



        kvm_inject_x86_mce(cenv, bank, status, mcg_status, addr, misc, flag);

    } else {

        qemu_inject_x86_mce(mon, cenv, bank, status, mcg_status, addr, misc,

                            flags);

        if (flags & MCE_INJECT_BROADCAST) {

            for (env = first_cpu; env != NULL; env = env->next_cpu) {

                if (cenv == env) {

                    continue;

                }

                qemu_inject_x86_mce(mon, env, 1,

                                    MCI_STATUS_VAL | MCI_STATUS_UC,

                                    MCG_STATUS_MCIP | MCG_STATUS_RIPV, 0, 0,

                                    flags);

            }

        }

    }

}
