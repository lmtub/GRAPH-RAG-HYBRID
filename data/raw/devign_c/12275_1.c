static void vfio_pci_load_rom(VFIODevice *vdev)

{

    struct vfio_region_info reg_info = {

        .argsz = sizeof(reg_info),

        .index = VFIO_PCI_ROM_REGION_INDEX

    };

    uint64_t size;

    off_t off = 0;

    size_t bytes;



    if (ioctl(vdev->fd, VFIO_DEVICE_GET_REGION_INFO, &reg_info)) {

        error_report("vfio: Error getting ROM info: %m");

        return;

    }



    DPRINTF("Device %04x:%02x:%02x.%x ROM:\n", vdev->host.domain,

            vdev->host.bus, vdev->host.slot, vdev->host.function);

    DPRINTF("  size: 0x%lx, offset: 0x%lx, flags: 0x%lx\n",

            (unsigned long)reg_info.size, (unsigned long)reg_info.offset,

            (unsigned long)reg_info.flags);



    vdev->rom_size = size = reg_info.size;

    vdev->rom_offset = reg_info.offset;



    if (!vdev->rom_size) {


        error_report("vfio-pci: Cannot read device rom at "

                    "%04x:%02x:%02x.%x\n",

                    vdev->host.domain, vdev->host.bus, vdev->host.slot,

                    vdev->host.function);

        error_printf("Device option ROM contents are probably invalid "

                    "(check dmesg).\nSkip option ROM probe with rombar=0, "

                    "or load from file with romfile=\n");

        return;

    }



    vdev->rom = g_malloc(size);

    memset(vdev->rom, 0xff, size);



    while (size) {

        bytes = pread(vdev->fd, vdev->rom + off, size, vdev->rom_offset + off);

        if (bytes == 0) {

            break;

        } else if (bytes > 0) {

            off += bytes;

            size -= bytes;

        } else {

            if (errno == EINTR || errno == EAGAIN) {

                continue;

            }

            error_report("vfio: Error reading device ROM: %m");

            break;

        }

    }

}