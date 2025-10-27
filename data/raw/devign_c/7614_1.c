static void arm_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    ARMCPU *cpu = ARM_CPU(obj);

    static bool inited;

    uint32_t Aff1, Aff0;



    cs->env_ptr = &cpu->env;

    cpu_exec_init(cs, &error_abort);

    cpu->cp_regs = g_hash_table_new_full(g_int_hash, g_int_equal,

                                         g_free, g_free);



    /* This cpu-id-to-MPIDR affinity is used only for TCG; KVM will override it.

     * We don't support setting cluster ID ([16..23]) (known as Aff2

     * in later ARM ARM versions), or any of the higher affinity level fields,

     * so these bits always RAZ.

     */

    Aff1 = cs->cpu_index / ARM_CPUS_PER_CLUSTER;

    Aff0 = cs->cpu_index % ARM_CPUS_PER_CLUSTER;

    cpu->mp_affinity = (Aff1 << ARM_AFF1_SHIFT) | Aff0;



#ifndef CONFIG_USER_ONLY

    /* Our inbound IRQ and FIQ lines */

    if (kvm_enabled()) {

        /* VIRQ and VFIQ are unused with KVM but we add them to maintain

         * the same interface as non-KVM CPUs.

         */

        qdev_init_gpio_in(DEVICE(cpu), arm_cpu_kvm_set_irq, 4);

    } else {

        qdev_init_gpio_in(DEVICE(cpu), arm_cpu_set_irq, 4);

    }



    cpu->gt_timer[GTIMER_PHYS] = timer_new(QEMU_CLOCK_VIRTUAL, GTIMER_SCALE,

                                                arm_gt_ptimer_cb, cpu);

    cpu->gt_timer[GTIMER_VIRT] = timer_new(QEMU_CLOCK_VIRTUAL, GTIMER_SCALE,

                                                arm_gt_vtimer_cb, cpu);

    cpu->gt_timer[GTIMER_HYP] = timer_new(QEMU_CLOCK_VIRTUAL, GTIMER_SCALE,

                                                arm_gt_htimer_cb, cpu);

    cpu->gt_timer[GTIMER_SEC] = timer_new(QEMU_CLOCK_VIRTUAL, GTIMER_SCALE,

                                                arm_gt_stimer_cb, cpu);

    qdev_init_gpio_out(DEVICE(cpu), cpu->gt_timer_outputs,

                       ARRAY_SIZE(cpu->gt_timer_outputs));

#endif



    /* DTB consumers generally don't in fact care what the 'compatible'

     * string is, so always provide some string and trust that a hypothetical

     * picky DTB consumer will also provide a helpful error message.

     */

    cpu->dtb_compatible = "qemu,unknown";

    cpu->psci_version = 1; /* By default assume PSCI v0.1 */

    cpu->kvm_target = QEMU_KVM_ARM_TARGET_NONE;



    if (tcg_enabled()) {

        cpu->psci_version = 2; /* TCG implements PSCI 0.2 */

        if (!inited) {

            inited = true;

            arm_translate_init();

        }

    }

}
