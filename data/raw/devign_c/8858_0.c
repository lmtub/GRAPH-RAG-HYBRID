static int vfio_load_rom(VFIODevice *vdev)

{

    uint64_t size = vdev->rom_size;

    char name[32];

    off_t off = 0, voff = vdev->rom_offset;

    ssize_t bytes;

    void *ptr;



    /* If loading ROM from file, pci handles it */

    if (vdev->pdev.romfile || !vdev->pdev.rom_bar || !size) {

        return 0;

    }



    DPRINTF("%s(%04x:%02x:%02x.%x)\n", __func__, vdev->host.domain,

            vdev->host.bus, vdev->host.slot, vdev->host.function);



    snprintf(name, sizeof(name), "vfio[%04x:%02x:%02x.%x].rom",

             vdev->host.domain, vdev->host.bus, vdev->host.slot,

             vdev->host.function);

    memory_region_init_ram(&vdev->pdev.rom, OBJECT(vdev), name, size);

    ptr = memory_region_get_ram_ptr(&vdev->pdev.rom);

    memset(ptr, 0xff, size);



    while (size) {

        bytes = pread(vdev->fd, ptr + off, size, voff + off);

        if (bytes == 0) {

            break; /* expect that we could get back less than the ROM BAR */

        } else if (bytes > 0) {

            off += bytes;

            size -= bytes;

        } else {

            if (errno == EINTR || errno == EAGAIN) {

                continue;

            }

            error_report("vfio: Error reading device ROM: %m");

            memory_region_destroy(&vdev->pdev.rom);

            return -errno;

        }

    }



    pci_register_bar(&vdev->pdev, PCI_ROM_SLOT, 0, &vdev->pdev.rom);

    vdev->pdev.has_rom = true;

    return 0;

}
