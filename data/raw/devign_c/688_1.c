void mips_r4k_init (ram_addr_t ram_size, int vga_ram_size,

                    const char *boot_device,

                    const char *kernel_filename, const char *kernel_cmdline,

                    const char *initrd_filename, const char *cpu_model)

{

    char buf[1024];

    unsigned long bios_offset;

    int bios_size;

    CPUState *env;

    RTCState *rtc_state;

    int i;

    qemu_irq *i8259;

    int index;

    BlockDriverState *hd[MAX_IDE_BUS * MAX_IDE_DEVS];



    /* init CPUs */

    if (cpu_model == NULL) {

#ifdef TARGET_MIPS64

        cpu_model = "R4000";

#else

        cpu_model = "24Kf";

#endif


    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");



    qemu_register_reset(main_cpu_reset, env);



    /* allocate RAM */







    cpu_register_physical_memory(0, ram_size, IO_MEM_RAM);



    if (!mips_qemu_iomemtype) {

        mips_qemu_iomemtype = cpu_register_io_memory(0, mips_qemu_read,

                                                     mips_qemu_write, NULL);


    cpu_register_physical_memory(0x1fbf0000, 0x10000, mips_qemu_iomemtype);



    /* Try to load a BIOS image. If this fails, we continue regardless,

       but initialize the hardware ourselves. When a kernel gets

       preloaded we also initialize the hardware, since the BIOS wasn't

       run. */

    bios_offset = ram_size + vga_ram_size;

    if (bios_name == NULL)

        bios_name = BIOS_FILENAME;

    snprintf(buf, sizeof(buf), "%s/%s", bios_dir, bios_name);

    bios_size = load_image(buf, phys_ram_base + bios_offset);

    if ((bios_size > 0) && (bios_size <= BIOS_SIZE)) {

	cpu_register_physical_memory(0x1fc00000,

				     BIOS_SIZE, bios_offset | IO_MEM_ROM);

    } else if ((index = drive_get_index(IF_PFLASH, 0, 0)) > -1) {

        uint32_t mips_rom = 0x00400000;

        cpu_register_physical_memory(0x1fc00000, mips_rom,

	                     qemu_ram_alloc(mips_rom) | IO_MEM_ROM);

        if (!pflash_cfi01_register(0x1fc00000, qemu_ram_alloc(mips_rom),

            drives_table[index].bdrv, sector_len, mips_rom / sector_len,

            4, 0, 0, 0, 0)) {

            fprintf(stderr, "qemu: Error registering flash memory.\n");



    else {

	/* not fatal */

        fprintf(stderr, "qemu: Warning, could not load MIPS bios '%s'\n",

		buf);




    if (kernel_filename) {

        loaderparams.ram_size = ram_size;

        loaderparams.kernel_filename = kernel_filename;

        loaderparams.kernel_cmdline = kernel_cmdline;

        loaderparams.initrd_filename = initrd_filename;

        load_kernel (env);




    /* Init CPU internal devices */

    cpu_mips_irq_init_cpu(env);

    cpu_mips_clock_init(env);



    /* The PIC is attached to the MIPS CPU INT0 pin */

    i8259 = i8259_init(env->irq[2]);



    rtc_state = rtc_init(0x70, i8259[8]);



    /* Register 64 KB of ISA IO space at 0x14000000 */

    isa_mmio_init(0x14000000, 0x00010000);

    isa_mem_base = 0x10000000;



    pit = pit_init(0x40, i8259[0]);



    for(i = 0; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            serial_init(serial_io[i], i8259[serial_irq[i]], 115200,

                        serial_hds[i]);





    isa_vga_init(phys_ram_base + ram_size, ram_size,

                 vga_ram_size);



    if (nd_table[0].vlan)

        isa_ne2000_init(0x300, i8259[9], &nd_table[0]);



    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");





    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        index = drive_get_index(IF_IDE, i / MAX_IDE_DEVS, i % MAX_IDE_DEVS);

        if (index != -1)

            hd[i] = drives_table[index].bdrv;

        else

            hd[i] = NULL;




    for(i = 0; i < MAX_IDE_BUS; i++)

        isa_ide_init(ide_iobase[i], ide_iobase2[i], i8259[ide_irq[i]],

                     hd[MAX_IDE_DEVS * i],

		     hd[MAX_IDE_DEVS * i + 1]);



    i8042_init(i8259[1], i8259[12], 0x60);
