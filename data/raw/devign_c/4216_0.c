static void pc_q35_init(QEMUMachineInitArgs *args)

{

    ram_addr_t below_4g_mem_size, above_4g_mem_size;

    Q35PCIHost *q35_host;

    PCIHostState *phb;

    PCIBus *host_bus;

    PCIDevice *lpc;

    BusState *idebus[MAX_SATA_PORTS];

    ISADevice *rtc_state;

    ISADevice *floppy;

    MemoryRegion *pci_memory;

    MemoryRegion *rom_memory;

    MemoryRegion *ram_memory;

    GSIState *gsi_state;

    ISABus *isa_bus;

    int pci_enabled = 1;

    qemu_irq *cpu_irq;

    qemu_irq *gsi;

    qemu_irq *i8259;

    int i;

    ICH9LPCState *ich9_lpc;

    PCIDevice *ahci;

    DeviceState *icc_bridge;

    PcGuestInfo *guest_info;



    icc_bridge = qdev_create(NULL, TYPE_ICC_BRIDGE);

    object_property_add_child(qdev_get_machine(), "icc-bridge",

                              OBJECT(icc_bridge), NULL);



    pc_cpus_init(args->cpu_model, icc_bridge);

    pc_acpi_init("q35-acpi-dsdt.aml");



    kvmclock_create();



    if (args->ram_size >= 0xb0000000) {

        above_4g_mem_size = args->ram_size - 0xb0000000;

        below_4g_mem_size = 0xb0000000;

    } else {

        above_4g_mem_size = 0;

        below_4g_mem_size = args->ram_size;

    }



    /* pci enabled */

    if (pci_enabled) {

        pci_memory = g_new(MemoryRegion, 1);

        memory_region_init(pci_memory, NULL, "pci", INT64_MAX);

        rom_memory = pci_memory;

    } else {

        pci_memory = NULL;

        rom_memory = get_system_memory();

    }



    guest_info = pc_guest_info_init(below_4g_mem_size, above_4g_mem_size);

    guest_info->has_pci_info = has_pci_info;

    guest_info->isapc_ram_fw = false;



    /* allocate ram and load rom/bios */

    if (!xen_enabled()) {

        pc_memory_init(get_system_memory(),

                       args->kernel_filename, args->kernel_cmdline,

                       args->initrd_filename,

                       below_4g_mem_size, above_4g_mem_size,

                       rom_memory, &ram_memory, guest_info);

    }



    /* irq lines */

    gsi_state = g_malloc0(sizeof(*gsi_state));

    if (kvm_irqchip_in_kernel()) {

        kvm_pc_setup_irq_routing(pci_enabled);

        gsi = qemu_allocate_irqs(kvm_pc_gsi_handler, gsi_state,

                                 GSI_NUM_PINS);

    } else {

        gsi = qemu_allocate_irqs(gsi_handler, gsi_state, GSI_NUM_PINS);

    }



    /* create pci host bus */

    q35_host = Q35_HOST_DEVICE(qdev_create(NULL, TYPE_Q35_HOST_DEVICE));



    object_property_add_child(qdev_get_machine(), "q35", OBJECT(q35_host), NULL);

    q35_host->mch.ram_memory = ram_memory;

    q35_host->mch.pci_address_space = pci_memory;

    q35_host->mch.system_memory = get_system_memory();

    q35_host->mch.address_space_io = get_system_io();

    q35_host->mch.below_4g_mem_size = below_4g_mem_size;

    q35_host->mch.above_4g_mem_size = above_4g_mem_size;

    q35_host->mch.guest_info = guest_info;

    /* pci */

    qdev_init_nofail(DEVICE(q35_host));

    phb = PCI_HOST_BRIDGE(q35_host);

    host_bus = phb->bus;

    /* create ISA bus */

    lpc = pci_create_simple_multifunction(host_bus, PCI_DEVFN(ICH9_LPC_DEV,

                                          ICH9_LPC_FUNC), true,

                                          TYPE_ICH9_LPC_DEVICE);

    ich9_lpc = ICH9_LPC_DEVICE(lpc);

    ich9_lpc->pic = gsi;

    ich9_lpc->ioapic = gsi_state->ioapic_irq;

    pci_bus_irqs(host_bus, ich9_lpc_set_irq, ich9_lpc_map_irq, ich9_lpc,

                 ICH9_LPC_NB_PIRQS);

    pci_bus_set_route_irq_fn(host_bus, ich9_route_intx_pin_to_irq);

    isa_bus = ich9_lpc->isa_bus;



    /*end early*/

    isa_bus_irqs(isa_bus, gsi);



    if (kvm_irqchip_in_kernel()) {

        i8259 = kvm_i8259_init(isa_bus);

    } else if (xen_enabled()) {

        i8259 = xen_interrupt_controller_init();

    } else {

        cpu_irq = pc_allocate_cpu_irq();

        i8259 = i8259_init(isa_bus, cpu_irq[0]);

    }



    for (i = 0; i < ISA_NUM_IRQS; i++) {

        gsi_state->i8259_irq[i] = i8259[i];

    }

    if (pci_enabled) {

        ioapic_init_gsi(gsi_state, NULL);

    }

    qdev_init_nofail(icc_bridge);



    pc_register_ferr_irq(gsi[13]);



    /* init basic PC hardware */

    pc_basic_device_init(isa_bus, gsi, &rtc_state, &floppy, false);



    /* connect pm stuff to lpc */

    ich9_lpc_pm_init(lpc);



    /* ahci and SATA device, for q35 1 ahci controller is built-in */

    ahci = pci_create_simple_multifunction(host_bus,

                                           PCI_DEVFN(ICH9_SATA1_DEV,

                                                     ICH9_SATA1_FUNC),

                                           true, "ich9-ahci");

    idebus[0] = qdev_get_child_bus(&ahci->qdev, "ide.0");

    idebus[1] = qdev_get_child_bus(&ahci->qdev, "ide.1");



    if (usb_enabled(false)) {

        /* Should we create 6 UHCI according to ich9 spec? */

        ehci_create_ich9_with_companions(host_bus, 0x1d);

    }



    /* TODO: Populate SPD eeprom data.  */

    smbus_eeprom_init(ich9_smb_init(host_bus,

                                    PCI_DEVFN(ICH9_SMB_DEV, ICH9_SMB_FUNC),

                                    0xb100),

                      8, NULL, 0);



    pc_cmos_init(below_4g_mem_size, above_4g_mem_size, args->boot_device,

                 floppy, idebus[0], idebus[1], rtc_state);



    /* the rest devices to which pci devfn is automatically assigned */

    pc_vga_init(isa_bus, host_bus);

    pc_nic_init(isa_bus, host_bus);

    if (pci_enabled) {

        pc_pci_device_init(host_bus);

    }



    if (has_pvpanic) {

        pvpanic_init(isa_bus);

    }

}
