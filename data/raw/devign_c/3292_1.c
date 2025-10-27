static int pci_add_option_rom(PCIDevice *pdev)

{

    int size;

    char *path;

    void *ptr;

    char name[32];



    if (!pdev->romfile)

        return 0;

    if (strlen(pdev->romfile) == 0)

        return 0;



    if (!pdev->rom_bar) {

        /*

         * Load rom via fw_cfg instead of creating a rom bar,

         * for 0.11 compatibility.

         */

        int class = pci_get_word(pdev->config + PCI_CLASS_DEVICE);

        if (class == 0x0300) {

            rom_add_vga(pdev->romfile);

        } else {

            rom_add_option(pdev->romfile);

        }

        return 0;

    }



    path = qemu_find_file(QEMU_FILE_TYPE_BIOS, pdev->romfile);

    if (path == NULL) {

        path = qemu_strdup(pdev->romfile);

    }



    size = get_image_size(path);

    if (size < 0) {

        error_report("%s: failed to find romfile \"%s\"",

                     __FUNCTION__, pdev->romfile);

        return -1;

    }

    if (size & (size - 1)) {

        size = 1 << qemu_fls(size);

    }



    if (pdev->qdev.info->vmsd)

        snprintf(name, sizeof(name), "%s.rom", pdev->qdev.info->vmsd->name);

    else

        snprintf(name, sizeof(name), "%s.rom", pdev->qdev.info->name);

    pdev->rom_offset = qemu_ram_alloc(&pdev->qdev, name, size);



    ptr = qemu_get_ram_ptr(pdev->rom_offset);

    load_image(path, ptr);

    qemu_free(path);



    pci_register_bar(pdev, PCI_ROM_SLOT, size,

                     0, pci_map_option_rom);



    return 0;

}
