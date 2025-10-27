void mips_malta_init (ram_addr_t ram_size, int vga_ram_size,

                      const char *boot_device,

                      const char *kernel_filename, const char *kernel_cmdline,

                      const char *initrd_filename, const char *cpu_model)

{

    char buf[1024];

    unsigned long bios_offset;

    target_long bios_size;

    int64_t kernel_entry;

    PCIBus *pci_bus;

    CPUState *env;

    RTCState *rtc_state;

    fdctrl_t *floppy_controller;

    MaltaFPGAState *malta_fpga;

    qemu_irq *i8259;

    int piix4_devfn;

    uint8_t *eeprom_buf;

    i2c_bus *smbus;

    int i;

    int index;

    BlockDriverState *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    BlockDriverState *fd[MAX_FD];

    int fl_idx = 0;

    int fl_sectors = 0;



    /* init CPUs */

    if (cpu_model == NULL) {

#ifdef TARGET_MIPS64

        cpu_model = "20Kc";

#else

        cpu_model = "24Kf";

#endif


    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");



    qemu_register_reset(main_cpu_reset, env);



    /* allocate RAM */







    cpu_register_physical_memory(0, ram_size, IO_MEM_RAM);



    /* Map the bios at two physical locations, as on the real board. */

    bios_offset = ram_size + vga_ram_size;

    cpu_register_physical_memory(0x1e000000LL,

                                 BIOS_SIZE, bios_offset | IO_MEM_ROM);

    cpu_register_physical_memory(0x1fc00000LL,

                                 BIOS_SIZE, bios_offset | IO_MEM_ROM);



    /* FPGA */

    malta_fpga = malta_fpga_init(0x1f000000LL, env->irq[2], serial_hds[2]);



    /* Load firmware in flash / BIOS unless we boot directly into a kernel. */

    if (kernel_filename) {

        /* Write a small bootloader to the flash location. */

        loaderparams.ram_size = ram_size;

        loaderparams.kernel_filename = kernel_filename;

        loaderparams.kernel_cmdline = kernel_cmdline;

        loaderparams.initrd_filename = initrd_filename;

        kernel_entry = load_kernel(env);

        env->CP0_Status &= ~((1 << CP0St_BEV) | (1 << CP0St_ERL));

        write_bootloader(env, bios_offset, kernel_entry);

    } else {

        index = drive_get_index(IF_PFLASH, 0, fl_idx);

        if (index != -1) {

            /* Load firmware from flash. */

            bios_size = 0x400000;

            fl_sectors = bios_size >> 16;

#ifdef DEBUG_BOARD_INIT

            printf("Register parallel flash %d size " TARGET_FMT_lx " at "

                   "offset %08lx addr %08llx '%s' %x\n",

                   fl_idx, bios_size, bios_offset, 0x1e000000LL,

                   bdrv_get_device_name(drives_table[index].bdrv), fl_sectors);

#endif

            pflash_cfi01_register(0x1e000000LL, bios_offset,

                                  drives_table[index].bdrv, 65536, fl_sectors,

                                  4, 0x0000, 0x0000, 0x0000, 0x0000);

            fl_idx++;

        } else {

            /* Load a BIOS image. */

            if (bios_name == NULL)

                bios_name = BIOS_FILENAME;

            snprintf(buf, sizeof(buf), "%s/%s", bios_dir, bios_name);

            bios_size = load_image(buf, phys_ram_base + bios_offset);

            if ((bios_size < 0 || bios_size > BIOS_SIZE) && !kernel_filename) {


                        "qemu: Could not load MIPS bios '%s', and no -kernel argument was specified\n",

                        buf);




        /* In little endian mode the 32bit words in the bios are swapped,

           a neat trick which allows bi-endian firmware. */

#ifndef TARGET_WORDS_BIGENDIAN

        {

            uint32_t *addr;

            for (addr = (uint32_t *)(phys_ram_base + bios_offset);

                 addr < (uint32_t *)(phys_ram_base + bios_offset + bios_size);

                 addr++) {

                *addr = bswap32(*addr);



#endif




    /* Board ID = 0x420 (Malta Board with CoreLV)

       XXX: theoretically 0x1e000010 should map to flash and 0x1fc00010 should

       map to the board ID. */

    stl_raw(phys_ram_base + bios_offset + 0x10, 0x00000420);



    /* Init internal devices */

    cpu_mips_irq_init_cpu(env);

    cpu_mips_clock_init(env);



    /* Interrupt controller */

    /* The 8259 is attached to the MIPS CPU INT0 pin, ie interrupt 2 */

    i8259 = i8259_init(env->irq[2]);



    /* Northbridge */

    pci_bus = pci_gt64120_init(i8259);



    /* Southbridge */



    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");





    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        index = drive_get_index(IF_IDE, i / MAX_IDE_DEVS, i % MAX_IDE_DEVS);

        if (index != -1)

            hd[i] = drives_table[index].bdrv;

        else

            hd[i] = NULL;




    piix4_devfn = piix4_init(pci_bus, 80);

    pci_piix4_ide_init(pci_bus, hd, piix4_devfn + 1, i8259);

    usb_uhci_piix4_init(pci_bus, piix4_devfn + 2);

    smbus = piix4_pm_init(pci_bus, piix4_devfn + 3, 0x1100, i8259[9]);

    eeprom_buf = qemu_mallocz(8 * 256); /* XXX: make this persistent */

    for (i = 0; i < 8; i++) {

        /* TODO: Populate SPD eeprom data.  */

        smbus_eeprom_device_init(smbus, 0x50 + i, eeprom_buf + (i * 256));


    pit = pit_init(0x40, i8259[0]);

    DMA_init(0);



    /* Super I/O */

    i8042_init(i8259[1], i8259[12], 0x60);

    rtc_state = rtc_init(0x70, i8259[8]);

    serial_init(0x3f8, i8259[4], 115200, serial_hds[0]);

    serial_init(0x2f8, i8259[3], 115200, serial_hds[1]);

    if (parallel_hds[0])

        parallel_init(0x378, i8259[7], parallel_hds[0]);

    for(i = 0; i < MAX_FD; i++) {

        index = drive_get_index(IF_FLOPPY, 0, i);

       if (index != -1)

           fd[i] = drives_table[index].bdrv;

       else

           fd[i] = NULL;


    floppy_controller = fdctrl_init(i8259[6], 2, 0, 0x3f0, fd);



    /* Sound card */

#ifdef HAS_AUDIO

    audio_init(pci_bus);

#endif



    /* Network card */

    network_init(pci_bus);



    /* Optional PCI video card */

    pci_cirrus_vga_init(pci_bus, phys_ram_base + ram_size,

                        ram_size, vga_ram_size);
