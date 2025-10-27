static void x86_cpu_realizefn(DeviceState *dev, Error **errp)

{

    CPUState *cs = CPU(dev);

    X86CPU *cpu = X86_CPU(dev);

    X86CPUClass *xcc = X86_CPU_GET_CLASS(dev);

    CPUX86State *env = &cpu->env;

    Error *local_err = NULL;

    static bool ht_warned;



    if (xcc->kvm_required && !kvm_enabled()) {

        char *name = x86_cpu_class_get_model_name(xcc);

        error_setg(&local_err, "CPU model '%s' requires KVM", name);

        g_free(name);

        goto out;

    }



    if (cpu->apic_id == UNASSIGNED_APIC_ID) {

        error_setg(errp, "apic-id property was not initialized properly");

        return;

    }



    x86_cpu_load_features(cpu, &local_err);

    if (local_err) {

        goto out;

    }



    if (x86_cpu_filter_features(cpu) &&

        (cpu->check_cpuid || cpu->enforce_cpuid)) {

        x86_cpu_report_filtered_features(cpu);

        if (cpu->enforce_cpuid) {

            error_setg(&local_err,

                       kvm_enabled() ?

                           "Host doesn't support requested features" :

                           "TCG doesn't support requested features");

            goto out;

        }

    }



    /* On AMD CPUs, some CPUID[8000_0001].EDX bits must match the bits on

     * CPUID[1].EDX.

     */

    if (IS_AMD_CPU(env)) {

        env->features[FEAT_8000_0001_EDX] &= ~CPUID_EXT2_AMD_ALIASES;

        env->features[FEAT_8000_0001_EDX] |= (env->features[FEAT_1_EDX]

           & CPUID_EXT2_AMD_ALIASES);

    }



    /* For 64bit systems think about the number of physical bits to present.

     * ideally this should be the same as the host; anything other than matching

     * the host can cause incorrect guest behaviour.

     * QEMU used to pick the magic value of 40 bits that corresponds to

     * consumer AMD devices but nothing else.

     */

    if (env->features[FEAT_8000_0001_EDX] & CPUID_EXT2_LM) {

        if (kvm_enabled()) {

            uint32_t host_phys_bits = x86_host_phys_bits();

            static bool warned;



            if (cpu->host_phys_bits) {

                /* The user asked for us to use the host physical bits */

                cpu->phys_bits = host_phys_bits;

            }



            /* Print a warning if the user set it to a value that's not the

             * host value.

             */

            if (cpu->phys_bits != host_phys_bits && cpu->phys_bits != 0 &&

                !warned) {

                error_report("Warning: Host physical bits (%u)"

                                 " does not match phys-bits property (%u)",

                                 host_phys_bits, cpu->phys_bits);

                warned = true;

            }



            if (cpu->phys_bits &&

                (cpu->phys_bits > TARGET_PHYS_ADDR_SPACE_BITS ||

                cpu->phys_bits < 32)) {

                error_setg(errp, "phys-bits should be between 32 and %u "

                                 " (but is %u)",

                                 TARGET_PHYS_ADDR_SPACE_BITS, cpu->phys_bits);

                return;

            }

        } else {

            if (cpu->phys_bits && cpu->phys_bits != TCG_PHYS_ADDR_BITS) {

                error_setg(errp, "TCG only supports phys-bits=%u",

                                  TCG_PHYS_ADDR_BITS);

                return;

            }

        }

        /* 0 means it was not explicitly set by the user (or by machine

         * compat_props or by the host code above). In this case, the default

         * is the value used by TCG (40).

         */

        if (cpu->phys_bits == 0) {

            cpu->phys_bits = TCG_PHYS_ADDR_BITS;

        }

    } else {

        /* For 32 bit systems don't use the user set value, but keep

         * phys_bits consistent with what we tell the guest.

         */

        if (cpu->phys_bits != 0) {

            error_setg(errp, "phys-bits is not user-configurable in 32 bit");

            return;

        }



        if (env->features[FEAT_1_EDX] & CPUID_PSE36) {

            cpu->phys_bits = 36;

        } else {

            cpu->phys_bits = 32;

        }

    }

    cpu_exec_init(cs, &error_abort);



    if (tcg_enabled()) {

        tcg_x86_init();

    }



#ifndef CONFIG_USER_ONLY

    qemu_register_reset(x86_cpu_machine_reset_cb, cpu);



    if (cpu->env.features[FEAT_1_EDX] & CPUID_APIC || smp_cpus > 1) {

        x86_cpu_apic_create(cpu, &local_err);

        if (local_err != NULL) {

            goto out;

        }

    }

#endif



    mce_init(cpu);



#ifndef CONFIG_USER_ONLY

    if (tcg_enabled()) {

        AddressSpace *newas = g_new(AddressSpace, 1);



        cpu->cpu_as_mem = g_new(MemoryRegion, 1);

        cpu->cpu_as_root = g_new(MemoryRegion, 1);



        /* Outer container... */

        memory_region_init(cpu->cpu_as_root, OBJECT(cpu), "memory", ~0ull);

        memory_region_set_enabled(cpu->cpu_as_root, true);



        /* ... with two regions inside: normal system memory with low

         * priority, and...

         */

        memory_region_init_alias(cpu->cpu_as_mem, OBJECT(cpu), "memory",

                                 get_system_memory(), 0, ~0ull);

        memory_region_add_subregion_overlap(cpu->cpu_as_root, 0, cpu->cpu_as_mem, 0);

        memory_region_set_enabled(cpu->cpu_as_mem, true);

        address_space_init(newas, cpu->cpu_as_root, "CPU");

        cs->num_ases = 1;

        cpu_address_space_init(cs, newas, 0);



        /* ... SMRAM with higher priority, linked from /machine/smram.  */

        cpu->machine_done.notify = x86_cpu_machine_done;

        qemu_add_machine_init_done_notifier(&cpu->machine_done);

    }

#endif



    qemu_init_vcpu(cs);



    /* Only Intel CPUs support hyperthreading. Even though QEMU fixes this

     * issue by adjusting CPUID_0000_0001_EBX and CPUID_8000_0008_ECX

     * based on inputs (sockets,cores,threads), it is still better to gives

     * users a warning.

     *

     * NOTE: the following code has to follow qemu_init_vcpu(). Otherwise

     * cs->nr_threads hasn't be populated yet and the checking is incorrect.

     */

    if (!IS_INTEL_CPU(env) && cs->nr_threads > 1 && !ht_warned) {

        error_report("AMD CPU doesn't support hyperthreading. Please configure"

                     " -smp options properly.");

        ht_warned = true;

    }



    x86_cpu_apic_realize(cpu, &local_err);

    if (local_err != NULL) {

        goto out;

    }

    cpu_reset(cs);



    xcc->parent_realize(dev, &local_err);



out:

    if (local_err != NULL) {

        error_propagate(errp, local_err);

        return;

    }

}
