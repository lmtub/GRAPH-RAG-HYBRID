void cpu_x86_inject_mce(CPUState *cenv, int bank, uint64_t status,

                        uint64_t mcg_status, uint64_t addr, uint64_t misc,

                        int broadcast)

{

    unsigned bank_num = cenv->mcg_cap & 0xff;

    CPUState *env;

    int flag = 0;



    if (bank >= bank_num || !(status & MCI_STATUS_VAL)) {

        return;

    }



    if (broadcast) {

        if (!cpu_x86_support_mca_broadcast(cenv)) {

            fprintf(stderr, "Current CPU does not support broadcast\n");

            return;

        }

    }



    if (kvm_enabled()) {

        if (broadcast) {

            flag |= MCE_BROADCAST;

        }



        kvm_inject_x86_mce(cenv, bank, status, mcg_status, addr, misc, flag);

    } else {

        qemu_inject_x86_mce(cenv, bank, status, mcg_status, addr, misc);

        if (broadcast) {

            for (env = first_cpu; env != NULL; env = env->next_cpu) {

                if (cenv == env) {

                    continue;

                }

                qemu_inject_x86_mce(env, 1, MCI_STATUS_VAL | MCI_STATUS_UC,

                                    MCG_STATUS_MCIP | MCG_STATUS_RIPV, 0, 0);

            }

        }

    }

}
