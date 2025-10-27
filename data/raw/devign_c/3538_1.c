PCIBus *typhoon_init(ram_addr_t ram_size, ISABus **isa_bus,

                     qemu_irq *p_rtc_irq,

                     AlphaCPU *cpus[4], pci_map_irq_fn sys_map_irq)

{

    const uint64_t MB = 1024 * 1024;

    const uint64_t GB = 1024 * MB;

    MemoryRegion *addr_space = get_system_memory();

    MemoryRegion *addr_space_io = get_system_io();

    DeviceState *dev;

    TyphoonState *s;

    PCIHostState *phb;

    PCIBus *b;

    int i;



    dev = qdev_create(NULL, TYPE_TYPHOON_PCI_HOST_BRIDGE);

    qdev_init_nofail(dev);



    s = TYPHOON_PCI_HOST_BRIDGE(dev);

    phb = PCI_HOST_BRIDGE(dev);



    /* Remember the CPUs so that we can deliver interrupts to them.  */

    for (i = 0; i < 4; i++) {

        AlphaCPU *cpu = cpus[i];

        s->cchip.cpu[i] = cpu;

        if (cpu != NULL) {

            CPUAlphaState *env = &cpu->env;

            env->alarm_timer = qemu_new_timer_ns(rtc_clock,

                                                 typhoon_alarm_timer,

                                                 (void *)((uintptr_t)s + i));

        }

    }



    *p_rtc_irq = *qemu_allocate_irqs(typhoon_set_timer_irq, s, 1);



    /* Main memory region, 0x00.0000.0000.  Real hardware supports 32GB,

       but the address space hole reserved at this point is 8TB.  */

    memory_region_init_ram(&s->ram_region, "ram", ram_size);

    vmstate_register_ram_global(&s->ram_region);

    memory_region_add_subregion(addr_space, 0, &s->ram_region);



    /* TIGbus, 0x801.0000.0000, 1GB.  */

    /* ??? The TIGbus is used for delivering interrupts, and access to

       the flash ROM.  I'm not sure that we need to implement it at all.  */



    /* Pchip0 CSRs, 0x801.8000.0000, 256MB.  */

    memory_region_init_io(&s->pchip.region, &pchip_ops, s, "pchip0", 256*MB);

    memory_region_add_subregion(addr_space, 0x80180000000ULL,

                                &s->pchip.region);



    /* Cchip CSRs, 0x801.A000.0000, 256MB.  */

    memory_region_init_io(&s->cchip.region, &cchip_ops, s, "cchip0", 256*MB);

    memory_region_add_subregion(addr_space, 0x801a0000000ULL,

                                &s->cchip.region);



    /* Dchip CSRs, 0x801.B000.0000, 256MB.  */

    memory_region_init_io(&s->dchip_region, &dchip_ops, s, "dchip0", 256*MB);

    memory_region_add_subregion(addr_space, 0x801b0000000ULL,

                                &s->dchip_region);



    /* Pchip0 PCI memory, 0x800.0000.0000, 4GB.  */

    memory_region_init(&s->pchip.reg_mem, "pci0-mem", 4*GB);

    memory_region_add_subregion(addr_space, 0x80000000000ULL,

                                &s->pchip.reg_mem);



    /* Pchip0 PCI I/O, 0x801.FC00.0000, 32MB.  */

    /* ??? Ideally we drop the "system" i/o space on the floor and give the

       PCI subsystem the full address space reserved by the chipset.

       We can't do that until the MEM and IO paths in memory.c are unified.  */

    memory_region_init_io(&s->pchip.reg_io, &alpha_pci_bw_io_ops, NULL,

                          "pci0-io", 32*MB);

    memory_region_add_subregion(addr_space, 0x801fc000000ULL,

                                &s->pchip.reg_io);



    b = pci_register_bus(dev, "pci",

                         typhoon_set_irq, sys_map_irq, s,

                         &s->pchip.reg_mem, addr_space_io, 0, 64);

    phb->bus = b;



    /* Pchip0 PCI special/interrupt acknowledge, 0x801.F800.0000, 64MB.  */

    memory_region_init_io(&s->pchip.reg_iack, &alpha_pci_iack_ops, b,

                          "pci0-iack", 64*MB);

    memory_region_add_subregion(addr_space, 0x801f8000000ULL,

                                &s->pchip.reg_iack);



    /* Pchip0 PCI configuration, 0x801.FE00.0000, 16MB.  */

    memory_region_init_io(&s->pchip.reg_conf, &alpha_pci_conf1_ops, b,

                          "pci0-conf", 16*MB);

    memory_region_add_subregion(addr_space, 0x801fe000000ULL,

                                &s->pchip.reg_conf);



    /* For the record, these are the mappings for the second PCI bus.

       We can get away with not implementing them because we indicate

       via the Cchip.CSC<PIP> bit that Pchip1 is not present.  */

    /* Pchip1 PCI memory, 0x802.0000.0000, 4GB.  */

    /* Pchip1 CSRs, 0x802.8000.0000, 256MB.  */

    /* Pchip1 PCI special/interrupt acknowledge, 0x802.F800.0000, 64MB.  */

    /* Pchip1 PCI I/O, 0x802.FC00.0000, 32MB.  */

    /* Pchip1 PCI configuration, 0x802.FE00.0000, 16MB.  */



    /* Init the ISA bus.  */

    /* ??? Technically there should be a cy82c693ub pci-isa bridge.  */

    {

        qemu_irq isa_pci_irq, *isa_irqs;



        *isa_bus = isa_bus_new(NULL, addr_space_io);

        isa_pci_irq = *qemu_allocate_irqs(typhoon_set_isa_irq, s, 1);

        isa_irqs = i8259_init(*isa_bus, isa_pci_irq);

        isa_bus_irqs(*isa_bus, isa_irqs);

    }



    return b;

}
