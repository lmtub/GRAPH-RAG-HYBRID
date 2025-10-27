void *slavio_intctl_init(target_phys_addr_t addr, target_phys_addr_t addrg,

                         const uint32_t *intbit_to_level,

                         qemu_irq **irq, qemu_irq **cpu_irq,

                         unsigned int cputimer)



{

    int slavio_intctl_io_memory, slavio_intctlm_io_memory, i;

    SLAVIO_INTCTLState *s;



    s = qemu_mallocz(sizeof(SLAVIO_INTCTLState));

    if (!s)

        return NULL;



    s->intbit_to_level = intbit_to_level;

    for (i = 0; i < MAX_CPUS; i++) {

	slavio_intctl_io_memory = cpu_register_io_memory(0, slavio_intctl_mem_read, slavio_intctl_mem_write, s);

	cpu_register_physical_memory(addr + i * TARGET_PAGE_SIZE, INTCTL_SIZE,

                                     slavio_intctl_io_memory);

    }



    slavio_intctlm_io_memory = cpu_register_io_memory(0, slavio_intctlm_mem_read, slavio_intctlm_mem_write, s);

    cpu_register_physical_memory(addrg, INTCTLM_SIZE, slavio_intctlm_io_memory);



    register_savevm("slavio_intctl", addr, 1, slavio_intctl_save, slavio_intctl_load, s);

    qemu_register_reset(slavio_intctl_reset, s);

    *irq = qemu_allocate_irqs(slavio_set_irq, s, 32);



    *cpu_irq = qemu_allocate_irqs(slavio_set_timer_irq_cpu, s, MAX_CPUS);

    s->cputimer_bit = 1 << s->intbit_to_level[cputimer];

    slavio_intctl_reset(s);

    return s;

}
