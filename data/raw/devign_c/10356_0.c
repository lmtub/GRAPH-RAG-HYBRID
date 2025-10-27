static int load_dtb(hwaddr addr, const struct arm_boot_info *binfo)

{

    void *fdt = NULL;

    int size, rc;

    uint32_t acells, scells;



    if (binfo->dtb_filename) {

        char *filename;

        filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, binfo->dtb_filename);

        if (!filename) {

            fprintf(stderr, "Couldn't open dtb file %s\n", binfo->dtb_filename);

            goto fail;

        }



        fdt = load_device_tree(filename, &size);

        if (!fdt) {

            fprintf(stderr, "Couldn't open dtb file %s\n", filename);

            g_free(filename);

            goto fail;

        }

        g_free(filename);

    } else if (binfo->get_dtb) {

        fdt = binfo->get_dtb(binfo, &size);

        if (!fdt) {

            fprintf(stderr, "Board was unable to create a dtb blob\n");

            goto fail;

        }

    }



    acells = qemu_fdt_getprop_cell(fdt, "/", "#address-cells");

    scells = qemu_fdt_getprop_cell(fdt, "/", "#size-cells");

    if (acells == 0 || scells == 0) {

        fprintf(stderr, "dtb file invalid (#address-cells or #size-cells 0)\n");

        goto fail;

    }



    if (scells < 2 && binfo->ram_size >= (1ULL << 32)) {

        /* This is user error so deserves a friendlier error message

         * than the failure of setprop_sized_cells would provide

         */

        fprintf(stderr, "qemu: dtb file not compatible with "

                "RAM size > 4GB\n");

        goto fail;

    }



    rc = qemu_fdt_setprop_sized_cells(fdt, "/memory", "reg",

                                      acells, binfo->loader_start,

                                      scells, binfo->ram_size);

    if (rc < 0) {

        fprintf(stderr, "couldn't set /memory/reg\n");

        goto fail;

    }



    if (binfo->kernel_cmdline && *binfo->kernel_cmdline) {

        rc = qemu_fdt_setprop_string(fdt, "/chosen", "bootargs",

                                     binfo->kernel_cmdline);

        if (rc < 0) {

            fprintf(stderr, "couldn't set /chosen/bootargs\n");

            goto fail;

        }

    }



    if (binfo->initrd_size) {

        rc = qemu_fdt_setprop_cell(fdt, "/chosen", "linux,initrd-start",

                                   binfo->initrd_start);

        if (rc < 0) {

            fprintf(stderr, "couldn't set /chosen/linux,initrd-start\n");

            goto fail;

        }



        rc = qemu_fdt_setprop_cell(fdt, "/chosen", "linux,initrd-end",

                                   binfo->initrd_start + binfo->initrd_size);

        if (rc < 0) {

            fprintf(stderr, "couldn't set /chosen/linux,initrd-end\n");

            goto fail;

        }

    }



    if (binfo->modify_dtb) {

        binfo->modify_dtb(binfo, fdt);

    }



    qemu_fdt_dumpdtb(fdt, size);



    /* Put the DTB into the memory map as a ROM image: this will ensure

     * the DTB is copied again upon reset, even if addr points into RAM.

     */

    rom_add_blob_fixed("dtb", fdt, size, addr);



    g_free(fdt);



    return 0;



fail:

    g_free(fdt);

    return -1;

}
