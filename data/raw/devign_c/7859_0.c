static void *bamboo_load_device_tree(target_phys_addr_t addr,

                                     uint32_t ramsize,

                                     target_phys_addr_t initrd_base,

                                     target_phys_addr_t initrd_size,

                                     const char *kernel_cmdline)

{

    void *fdt = NULL;

#ifdef CONFIG_FDT

    uint32_t mem_reg_property[] = { 0, 0, ramsize };

    char *filename;

    int fdt_size;

    int ret;



    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, BINARY_DEVICE_TREE_FILE);

    if (!filename) {

        goto out;

    }

    fdt = load_device_tree(filename, &fdt_size);

    qemu_free(filename);

    if (fdt == NULL) {

        goto out;

    }



    /* Manipulate device tree in memory. */



    ret = qemu_devtree_setprop(fdt, "/memory", "reg", mem_reg_property,

                               sizeof(mem_reg_property));

    if (ret < 0)

        fprintf(stderr, "couldn't set /memory/reg\n");



    ret = qemu_devtree_setprop_cell(fdt, "/chosen", "linux,initrd-start",

                                    initrd_base);

    if (ret < 0)

        fprintf(stderr, "couldn't set /chosen/linux,initrd-start\n");



    ret = qemu_devtree_setprop_cell(fdt, "/chosen", "linux,initrd-end",

                                    (initrd_base + initrd_size));

    if (ret < 0)

        fprintf(stderr, "couldn't set /chosen/linux,initrd-end\n");



    ret = qemu_devtree_setprop_string(fdt, "/chosen", "bootargs",

                                      kernel_cmdline);

    if (ret < 0)

        fprintf(stderr, "couldn't set /chosen/bootargs\n");



    if (kvm_enabled())

        kvmppc_fdt_update(fdt);



    cpu_physical_memory_write (addr, (void *)fdt, fdt_size);



out:

#endif



    return fdt;

}
