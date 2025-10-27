int apic_init(CPUState *env)

{

    APICState *s;



    if (last_apic_idx >= MAX_APICS)

        return -1;

    s = qemu_mallocz(sizeof(APICState));

    env->apic_state = s;

    s->idx = last_apic_idx++;

    s->id = env->cpuid_apic_id;

    s->cpu_env = env;



    apic_reset(s);

    msix_supported = 1;



    /* XXX: mapping more APICs at the same memory location */

    if (apic_io_memory == 0) {

        /* NOTE: the APIC is directly connected to the CPU - it is not

           on the global memory bus. */

        apic_io_memory = cpu_register_io_memory(apic_mem_read,

                                                apic_mem_write, NULL);

        /* XXX: what if the base changes? */

        cpu_register_physical_memory(MSI_ADDR_BASE, MSI_ADDR_SIZE,

                                     apic_io_memory);

    }

    s->timer = qemu_new_timer(vm_clock, apic_timer, s);



    vmstate_register(s->idx, &vmstate_apic, s);

    qemu_register_reset(apic_reset, s);



    local_apics[s->idx] = s;

    return 0;

}
