qemu_irq *armv7m_init(int flash_size, int sram_size,

                      const char *kernel_filename, const char *cpu_model)

{

    CPUState *env;

    DeviceState *nvic;

    /* FIXME: make this local state.  */

    static qemu_irq pic[64];

    qemu_irq *cpu_pic;

    uint32_t pc;

    int image_size;

    uint64_t entry;

    uint64_t lowaddr;

    int i;

    int big_endian;



    flash_size *= 1024;

    sram_size *= 1024;



    if (!cpu_model)

	cpu_model = "cortex-m3";

    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }



#if 0

    /* > 32Mb SRAM gets complicated because it overlaps the bitband area.

       We don't have proper commandline options, so allocate half of memory

       as SRAM, up to a maximum of 32Mb, and the rest as code.  */

    if (ram_size > (512 + 32) * 1024 * 1024)

        ram_size = (512 + 32) * 1024 * 1024;

    sram_size = (ram_size / 2) & TARGET_PAGE_MASK;

    if (sram_size > 32 * 1024 * 1024)

        sram_size = 32 * 1024 * 1024;

    code_size = ram_size - sram_size;

#endif



    /* Flash programming is done via the SCU, so pretend it is ROM.  */

    cpu_register_physical_memory(0, flash_size,

                                 qemu_ram_alloc(flash_size) | IO_MEM_ROM);

    cpu_register_physical_memory(0x20000000, sram_size,

                                 qemu_ram_alloc(sram_size) | IO_MEM_RAM);

    armv7m_bitband_init();



    nvic = qdev_create(NULL, "armv7m_nvic");

    env->v7m.nvic = nvic;

    qdev_init(nvic);

    cpu_pic = arm_pic_init_cpu(env);

    sysbus_connect_irq(sysbus_from_qdev(nvic), 0, cpu_pic[ARM_PIC_CPU_IRQ]);

    for (i = 0; i < 64; i++) {

        pic[i] = qdev_get_gpio_in(nvic, i);

    }



#ifdef TARGET_WORDS_BIGENDIAN

    big_endian = 1;

#else

    big_endian = 0;

#endif



    image_size = load_elf(kernel_filename, 0, &entry, &lowaddr, NULL,

                          big_endian, ELF_MACHINE, 1);

    if (image_size < 0) {

        image_size = load_image_targphys(kernel_filename, 0, flash_size);

	lowaddr = 0;

    }

    if (image_size < 0) {

        fprintf(stderr, "qemu: could not load kernel '%s'\n",

                kernel_filename);

        exit(1);

    }



    /* If the image was loaded at address zero then assume it is a

       regular ROM image and perform the normal CPU reset sequence.

       Otherwise jump directly to the entry point.  */

    if (lowaddr == 0) {

	env->regs[13] = ldl_phys(0);

	pc = ldl_phys(4);

    } else {

	pc = entry;

    }

    env->thumb = pc & 1;

    env->regs[15] = pc & ~1;



    /* Hack to map an additional page of ram at the top of the address

       space.  This stops qemu complaining about executing code outside RAM

       when returning from an exception.  */

    cpu_register_physical_memory(0xfffff000, 0x1000,

                                 qemu_ram_alloc(0x1000) | IO_MEM_RAM);



    return pic;

}
