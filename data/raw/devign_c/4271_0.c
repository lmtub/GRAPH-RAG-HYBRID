static void s390_init(ram_addr_t my_ram_size,

                      const char *boot_device,

                      const char *kernel_filename,

                      const char *kernel_cmdline,

                      const char *initrd_filename,

                      const char *cpu_model)

{

    CPUS390XState *env = NULL;

    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    ram_addr_t kernel_size = 0;

    ram_addr_t initrd_offset;

    ram_addr_t initrd_size = 0;

    int shift = 0;

    uint8_t *storage_keys;

    void *virtio_region;

    target_phys_addr_t virtio_region_len;

    target_phys_addr_t virtio_region_start;

    int i;



    /* s390x ram size detection needs a 16bit multiplier + an increment. So

       guests > 64GB can be specified in 2MB steps etc. */

    while ((my_ram_size >> (20 + shift)) > 65535) {

        shift++;

    }

    my_ram_size = my_ram_size >> (20 + shift) << (20 + shift);



    /* lets propagate the changed ram size into the global variable. */

    ram_size = my_ram_size;



    /* get a BUS */

    s390_bus = s390_virtio_bus_init(&my_ram_size);



    /* allocate RAM */

    memory_region_init_ram(ram, "s390.ram", my_ram_size);

    vmstate_register_ram_global(ram);

    memory_region_add_subregion(sysmem, 0, ram);



    /* clear virtio region */

    virtio_region_len = my_ram_size - ram_size;

    virtio_region_start = ram_size;

    virtio_region = cpu_physical_memory_map(virtio_region_start,

                                            &virtio_region_len, true);

    memset(virtio_region, 0, virtio_region_len);

    cpu_physical_memory_unmap(virtio_region, virtio_region_len, 1,

                              virtio_region_len);



    /* allocate storage keys */

    storage_keys = g_malloc0(my_ram_size / TARGET_PAGE_SIZE);



    /* init CPUs */

    if (cpu_model == NULL) {

        cpu_model = "host";

    }



    ipi_states = g_malloc(sizeof(CPUS390XState *) * smp_cpus);



    for (i = 0; i < smp_cpus; i++) {

        CPUS390XState *tmp_env;



        tmp_env = cpu_init(cpu_model);

        if (!env) {

            env = tmp_env;

        }

        ipi_states[i] = tmp_env;

        tmp_env->halted = 1;

        tmp_env->exception_index = EXCP_HLT;

        tmp_env->storage_keys = storage_keys;

    }



    /* One CPU has to run */

    s390_add_running_cpu(env);



    if (kernel_filename) {



        kernel_size = load_elf(kernel_filename, NULL, NULL, NULL, NULL,

                               NULL, 1, ELF_MACHINE, 0);

        if (kernel_size == -1UL) {

            kernel_size = load_image_targphys(kernel_filename, 0, ram_size);

        }

        /*

         * we can not rely on the ELF entry point, since up to 3.2 this

         * value was 0x800 (the SALIPL loader) and it wont work. For

         * all (Linux) cases 0x10000 (KERN_IMAGE_START) should be fine.

         */

        env->psw.addr = KERN_IMAGE_START;

        env->psw.mask = 0x0000000180000000ULL;

    } else {

        ram_addr_t bios_size = 0;

        char *bios_filename;



        /* Load zipl bootloader */

        if (bios_name == NULL) {

            bios_name = ZIPL_FILENAME;

        }



        bios_filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

        bios_size = load_image_targphys(bios_filename, ZIPL_LOAD_ADDR, 4096);

        g_free(bios_filename);



        if ((long)bios_size < 0) {

            hw_error("could not load bootloader '%s'\n", bios_name);

        }



        if (bios_size > 4096) {

            hw_error("stage1 bootloader is > 4k\n");

        }



        env->psw.addr = ZIPL_START;

        env->psw.mask = 0x0000000180000000ULL;

    }



    if (initrd_filename) {

        initrd_offset = INITRD_START;

        while (kernel_size + 0x100000 > initrd_offset) {

            initrd_offset += 0x100000;

        }

        initrd_size = load_image_targphys(initrd_filename, initrd_offset,

                                          ram_size - initrd_offset);

        /* we have to overwrite values in the kernel image, which are "rom" */

        memcpy(rom_ptr(INITRD_PARM_START), &initrd_offset, 8);

        memcpy(rom_ptr(INITRD_PARM_SIZE), &initrd_size, 8);

    }



    if (kernel_cmdline) {

        /* we have to overwrite values in the kernel image, which are "rom" */

        memcpy(rom_ptr(KERN_PARM_AREA), kernel_cmdline,

               strlen(kernel_cmdline) + 1);

    }



    /* Create VirtIO network adapters */

    for(i = 0; i < nb_nics; i++) {

        NICInfo *nd = &nd_table[i];

        DeviceState *dev;



        if (!nd->model) {

            nd->model = g_strdup("virtio");

        }



        if (strcmp(nd->model, "virtio")) {

            fprintf(stderr, "S390 only supports VirtIO nics\n");

            exit(1);

        }



        dev = qdev_create((BusState *)s390_bus, "virtio-net-s390");

        qdev_set_nic_properties(dev, nd);

        qdev_init_nofail(dev);

    }



    /* Create VirtIO disk drives */

    for(i = 0; i < MAX_BLK_DEVS; i++) {

        DriveInfo *dinfo;

        DeviceState *dev;



        dinfo = drive_get(IF_IDE, 0, i);

        if (!dinfo) {

            continue;

        }



        dev = qdev_create((BusState *)s390_bus, "virtio-blk-s390");

        qdev_prop_set_drive_nofail(dev, "drive", dinfo->bdrv);

        qdev_init_nofail(dev);

    }

}
