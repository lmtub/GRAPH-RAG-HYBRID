static void spapr_machine_init(MachineState *machine)
{
    sPAPRMachineState *spapr = SPAPR_MACHINE(machine);
    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(machine);
    const char *kernel_filename = machine->kernel_filename;
    const char *initrd_filename = machine->initrd_filename;
    PCIHostState *phb;
    int i;
    MemoryRegion *sysmem = get_system_memory();
    MemoryRegion *ram = g_new(MemoryRegion, 1);
    MemoryRegion *rma_region;
    void *rma = NULL;
    hwaddr rma_alloc_size;
    hwaddr node0_size = spapr_node0_size(machine);
    long load_limit, fw_size;
    char *filename;
    Error *resize_hpt_err = NULL;
    msi_nonbroken = true;
    QLIST_INIT(&spapr->phbs);
    QTAILQ_INIT(&spapr->pending_dimm_unplugs);
    /* Check HPT resizing availability */
    kvmppc_check_papr_resize_hpt(&resize_hpt_err);
    if (spapr->resize_hpt == SPAPR_RESIZE_HPT_DEFAULT) {
        /*
         * If the user explicitly requested a mode we should either
         * supply it, or fail completely (which we do below).  But if
         * it's not set explicitly, we reset our mode to something
         * that works
         */
        if (resize_hpt_err) {
            spapr->resize_hpt = SPAPR_RESIZE_HPT_DISABLED;
            error_free(resize_hpt_err);
            resize_hpt_err = NULL;
        } else {
            spapr->resize_hpt = smc->resize_hpt_default;
        }
    }
    assert(spapr->resize_hpt != SPAPR_RESIZE_HPT_DEFAULT);
    if ((spapr->resize_hpt != SPAPR_RESIZE_HPT_DISABLED) && resize_hpt_err) {
        /*
         * User requested HPT resize, but this host can't supply it.  Bail out
         */
        error_report_err(resize_hpt_err);
        exit(1);
    }
    /* Allocate RMA if necessary */
    rma_alloc_size = kvmppc_alloc_rma(&rma);
    if (rma_alloc_size == -1) {
        error_report("Unable to create RMA");
        exit(1);
    }
    if (rma_alloc_size && (rma_alloc_size < node0_size)) {
        spapr->rma_size = rma_alloc_size;
    } else {
        spapr->rma_size = node0_size;
        /* With KVM, we don't actually know whether KVM supports an
         * unbounded RMA (PR KVM) or is limited by the hash table size
         * (HV KVM using VRMA), so we always assume the latter
         *
         * In that case, we also limit the initial allocations for RTAS
         * etc... to 256M since we have no way to know what the VRMA size
         * is going to be as it depends on the size of the hash table
         * isn't determined yet.
         */
        if (kvm_enabled()) {
            spapr->vrma_adjust = 1;
            spapr->rma_size = MIN(spapr->rma_size, 0x10000000);
        }
        /* Actually we don't support unbounded RMA anymore since we
         * added proper emulation of HV mode. The max we can get is
         * 16G which also happens to be what we configure for PAPR
         * mode so make sure we don't do anything bigger than that
         */
        spapr->rma_size = MIN(spapr->rma_size, 0x400000000ull);
    }
    if (spapr->rma_size > node0_size) {
        error_report("Numa node 0 has to span the RMA (%#08"HWADDR_PRIx")",
                     spapr->rma_size);
        exit(1);
    }
    /* Setup a load limit for the ramdisk leaving room for SLOF and FDT */
    load_limit = MIN(spapr->rma_size, RTAS_MAX_ADDR) - FW_OVERHEAD;
    /* Set up Interrupt Controller before we create the VCPUs */
    xics_system_init(machine, XICS_IRQS_SPAPR, &error_fatal);
    /* Set up containers for ibm,client-architecture-support negotiated options
     */
    spapr->ov5 = spapr_ovec_new();
    spapr->ov5_cas = spapr_ovec_new();
    if (smc->dr_lmb_enabled) {
        spapr_ovec_set(spapr->ov5, OV5_DRCONF_MEMORY);
        spapr_validate_node_memory(machine, &error_fatal);
    }
    spapr_ovec_set(spapr->ov5, OV5_FORM1_AFFINITY);
    if (!kvm_enabled() || kvmppc_has_cap_mmu_radix()) {
        /* KVM and TCG always allow GTSE with radix... */
        spapr_ovec_set(spapr->ov5, OV5_MMU_RADIX_GTSE);
    }
    /* ... but not with hash (currently). */
    /* advertise support for dedicated HP event source to guests */
    if (spapr->use_hotplug_event_source) {
        spapr_ovec_set(spapr->ov5, OV5_HP_EVT);
    }
    /* advertise support for HPT resizing */
    if (spapr->resize_hpt != SPAPR_RESIZE_HPT_DISABLED) {
        spapr_ovec_set(spapr->ov5, OV5_HPT_RESIZE);
    }
    /* init CPUs */
    spapr_set_vsmt_mode(spapr, &error_fatal);
    spapr_init_cpus(spapr);
    if (kvm_enabled()) {
        /* Enable H_LOGICAL_CI_* so SLOF can talk to in-kernel devices */
        kvmppc_enable_logical_ci_hcalls();
        kvmppc_enable_set_mode_hcall();
        /* H_CLEAR_MOD/_REF are mandatory in PAPR, but off by default */
        kvmppc_enable_clear_ref_mod_hcalls();
    }
    /* allocate RAM */
    memory_region_allocate_system_memory(ram, NULL, "ppc_spapr.ram",
                                         machine->ram_size);
    memory_region_add_subregion(sysmem, 0, ram);
    if (rma_alloc_size && rma) {
        rma_region = g_new(MemoryRegion, 1);
        memory_region_init_ram_ptr(rma_region, NULL, "ppc_spapr.rma",
                                   rma_alloc_size, rma);
        vmstate_register_ram_global(rma_region);
        memory_region_add_subregion(sysmem, 0, rma_region);
    }
    /* initialize hotplug memory address space */
    if (machine->ram_size < machine->maxram_size) {
        ram_addr_t hotplug_mem_size = machine->maxram_size - machine->ram_size;
        /*
         * Limit the number of hotpluggable memory slots to half the number
         * slots that KVM supports, leaving the other half for PCI and other
         * devices. However ensure that number of slots doesn't drop below 32.
         */
        int max_memslots = kvm_enabled() ? kvm_get_max_memslots() / 2 :
                           SPAPR_MAX_RAM_SLOTS;
        if (max_memslots < SPAPR_MAX_RAM_SLOTS) {
            max_memslots = SPAPR_MAX_RAM_SLOTS;
        }
        if (machine->ram_slots > max_memslots) {
            error_report("Specified number of memory slots %"
                         PRIu64" exceeds max supported %d",
                         machine->ram_slots, max_memslots);
            exit(1);
        }
        spapr->hotplug_memory.base = ROUND_UP(machine->ram_size,
                                              SPAPR_HOTPLUG_MEM_ALIGN);
        memory_region_init(&spapr->hotplug_memory.mr, OBJECT(spapr),
                           "hotplug-memory", hotplug_mem_size);
        memory_region_add_subregion(sysmem, spapr->hotplug_memory.base,
                                    &spapr->hotplug_memory.mr);
    }
    if (smc->dr_lmb_enabled) {
        spapr_create_lmb_dr_connectors(spapr);
    }
    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, "spapr-rtas.bin");
    if (!filename) {
        error_report("Could not find LPAR rtas '%s'", "spapr-rtas.bin");
        exit(1);
    }
    spapr->rtas_size = get_image_size(filename);
    if (spapr->rtas_size < 0) {
        error_report("Could not get size of LPAR rtas '%s'", filename);
        exit(1);
    }
    spapr->rtas_blob = g_malloc(spapr->rtas_size);
    if (load_image_size(filename, spapr->rtas_blob, spapr->rtas_size) < 0) {
        error_report("Could not load LPAR rtas '%s'", filename);
        exit(1);
    }
    if (spapr->rtas_size > RTAS_MAX_SIZE) {
        error_report("RTAS too big ! 0x%zx bytes (max is 0x%x)",
                     (size_t)spapr->rtas_size, RTAS_MAX_SIZE);
        exit(1);
    }
    g_free(filename);
    /* Set up RTAS event infrastructure */
    spapr_events_init(spapr);
    /* Set up the RTC RTAS interfaces */
    spapr_rtc_create(spapr);
    /* Set up VIO bus */
    spapr->vio_bus = spapr_vio_bus_init();
    for (i = 0; i < MAX_SERIAL_PORTS; i++) {
        if (serial_hds[i]) {
            spapr_vty_create(spapr->vio_bus, serial_hds[i]);
        }
    }
    /* We always have at least the nvram device on VIO */
    spapr_create_nvram(spapr);
    /* Set up PCI */
    spapr_pci_rtas_init();
    phb = spapr_create_phb(spapr, 0);
    for (i = 0; i < nb_nics; i++) {
        NICInfo *nd = &nd_table[i];
        if (!nd->model) {
            nd->model = g_strdup("ibmveth");
        }
        if (strcmp(nd->model, "ibmveth") == 0) {
            spapr_vlan_create(spapr->vio_bus, nd);
        } else {
            pci_nic_init_nofail(&nd_table[i], phb->bus, nd->model, NULL);
        }
    }
    for (i = 0; i <= drive_get_max_bus(IF_SCSI); i++) {
        spapr_vscsi_create(spapr->vio_bus);
    }
    /* Graphics */
    if (spapr_vga_init(phb->bus, &error_fatal)) {
        spapr->has_graphics = true;
        machine->usb |= defaults_enabled() && !machine->usb_disabled;
    }
    if (machine->usb) {
        if (smc->use_ohci_by_default) {
            pci_create_simple(phb->bus, -1, "pci-ohci");
        } else {
            pci_create_simple(phb->bus, -1, "nec-usb-xhci");
        }
        if (spapr->has_graphics) {
            USBBus *usb_bus = usb_bus_find(-1);
            usb_create_simple(usb_bus, "usb-kbd");
            usb_create_simple(usb_bus, "usb-mouse");
        }
    }
    if (spapr->rma_size < (MIN_RMA_SLOF << 20)) {
        error_report(
            "pSeries SLOF firmware requires >= %ldM guest RMA (Real Mode Area memory)",
            MIN_RMA_SLOF);
        exit(1);
    }
    if (kernel_filename) {
        uint64_t lowaddr = 0;
        spapr->kernel_size = load_elf(kernel_filename, translate_kernel_address,
                                      NULL, NULL, &lowaddr, NULL, 1,
                                      PPC_ELF_MACHINE, 0, 0);
        if (spapr->kernel_size == ELF_LOAD_WRONG_ENDIAN) {
            spapr->kernel_size = load_elf(kernel_filename,
                                          translate_kernel_address, NULL, NULL,
                                          &lowaddr, NULL, 0, PPC_ELF_MACHINE,
                                          0, 0);
            spapr->kernel_le = spapr->kernel_size > 0;
        }
        if (spapr->kernel_size < 0) {
            error_report("error loading %s: %s", kernel_filename,
                         load_elf_strerror(spapr->kernel_size));
            exit(1);
        }
        /* load initrd */
        if (initrd_filename) {
            /* Try to locate the initrd in the gap between the kernel
             * and the firmware. Add a bit of space just in case
             */
            spapr->initrd_base = (KERNEL_LOAD_ADDR + spapr->kernel_size
                                  + 0x1ffff) & ~0xffff;
            spapr->initrd_size = load_image_targphys(initrd_filename,
                                                     spapr->initrd_base,
                                                     load_limit
                                                     - spapr->initrd_base);
            if (spapr->initrd_size < 0) {
                error_report("could not load initial ram disk '%s'",
                             initrd_filename);
                exit(1);
            }
        }
    }
    if (bios_name == NULL) {
        bios_name = FW_FILE_NAME;
    }
    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);
    if (!filename) {
        error_report("Could not find LPAR firmware '%s'", bios_name);
        exit(1);
    }
    fw_size = load_image_targphys(filename, 0, FW_MAX_SIZE);
    if (fw_size <= 0) {
        error_report("Could not load LPAR firmware '%s'", filename);
        exit(1);
    }
    g_free(filename);
    /* FIXME: Should register things through the MachineState's qdev
     * interface, this is a legacy from the sPAPREnvironment structure
     * which predated MachineState but had a similar function */
    vmstate_register(NULL, 0, &vmstate_spapr, spapr);
    register_savevm_live(NULL, "spapr/htab", -1, 1,
                         &savevm_htab_handlers, spapr);
    qemu_register_boot_set(spapr_boot_set, spapr);
    if (kvm_enabled()) {
        /* to stop and start vmclock */
        qemu_add_vm_change_state_handler(cpu_ppc_clock_vm_state_change,
                                         &spapr->tb);
        kvmppc_spapr_enable_inkernel_multitce();
    }
}