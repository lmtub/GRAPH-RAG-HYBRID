static void vfio_pci_size_rom(VFIODevice *vdev)

{

    uint32_t orig, size = cpu_to_le32((uint32_t)PCI_ROM_ADDRESS_MASK);

    off_t offset = vdev->config_offset + PCI_ROM_ADDRESS;

    char name[32];



    if (vdev->pdev.romfile || !vdev->pdev.rom_bar) {

        return;

    }



    /*

     * Use the same size ROM BAR as the physical device.  The contents

     * will get filled in later when the guest tries to read it.

     */

    if (pread(vdev->fd, &orig, 4, offset) != 4 ||

        pwrite(vdev->fd, &size, 4, offset) != 4 ||

        pread(vdev->fd, &size, 4, offset) != 4 ||

        pwrite(vdev->fd, &orig, 4, offset) != 4) {

        error_report("%s(%04x:%02x:%02x.%x) failed: %m",

                     __func__, vdev->host.domain, vdev->host.bus,

                     vdev->host.slot, vdev->host.function);

        return;

    }



    size = ~(le32_to_cpu(size) & PCI_ROM_ADDRESS_MASK) + 1;



    if (!size) {

        return;

    }



    DPRINTF("%04x:%02x:%02x.%x ROM size 0x%x\n", vdev->host.domain,

            vdev->host.bus, vdev->host.slot, vdev->host.function, size);



    snprintf(name, sizeof(name), "vfio[%04x:%02x:%02x.%x].rom",

             vdev->host.domain, vdev->host.bus, vdev->host.slot,

             vdev->host.function);



    memory_region_init_io(&vdev->pdev.rom, OBJECT(vdev),

                          &vfio_rom_ops, vdev, name, size);



    pci_register_bar(&vdev->pdev, PCI_ROM_SLOT,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &vdev->pdev.rom);



    vdev->pdev.has_rom = true;


}