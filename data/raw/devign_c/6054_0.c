static void kvmclock_vm_state_change(void *opaque, int running,

                                     RunState state)

{

    KVMClockState *s = opaque;

    CPUState *cpu;

    int cap_clock_ctrl = kvm_check_extension(kvm_state, KVM_CAP_KVMCLOCK_CTRL);

    int ret;



    if (running) {

        struct kvm_clock_data data = {};

        uint64_t time_at_migration = kvmclock_current_nsec(s);



        s->clock_valid = false;



        /* We can't rely on the migrated clock value, just discard it */

        if (time_at_migration) {

            s->clock = time_at_migration;

        }



        data.clock = s->clock;

        ret = kvm_vm_ioctl(kvm_state, KVM_SET_CLOCK, &data);

        if (ret < 0) {

            fprintf(stderr, "KVM_SET_CLOCK failed: %s\n", strerror(ret));

            abort();

        }



        if (!cap_clock_ctrl) {

            return;

        }

        CPU_FOREACH(cpu) {

            ret = kvm_vcpu_ioctl(cpu, KVM_KVMCLOCK_CTRL, 0);

            if (ret) {

                if (ret != -EINVAL) {

                    fprintf(stderr, "%s: %s\n", __func__, strerror(-ret));

                }

                return;

            }

        }

    } else {

        struct kvm_clock_data data;

        int ret;



        if (s->clock_valid) {

            return;

        }



        kvm_synchronize_all_tsc();



        ret = kvm_vm_ioctl(kvm_state, KVM_GET_CLOCK, &data);

        if (ret < 0) {

            fprintf(stderr, "KVM_GET_CLOCK failed: %s\n", strerror(ret));

            abort();

        }

        s->clock = data.clock;



        /*

         * If the VM is stopped, declare the clock state valid to

         * avoid re-reading it on next vmsave (which would return

         * a different value). Will be reset when the VM is continued.

         */

        s->clock_valid = true;

    }

}
