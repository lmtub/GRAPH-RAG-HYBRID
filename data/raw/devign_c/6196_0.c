static void vfio_pci_load_rom(VFIOPCIDevice *vdev)

{

    struct vfio_region_info reg_info = {

        .argsz = sizeof(reg_info),

        .index = VFIO_PCI_ROM_REGION_INDEX

    };

    uint64_t size;

    off_t off = 0;

    size_t bytes;



    if (ioctl(vdev->vbasedev.fd, VFIO_DEVICE_GET_REGION_INFO, &reg_info)) {

        error_report("vfio: Error getting ROM info: %m");

        return;

    }



    trace_vfio_pci_load_rom(vdev->vbasedev.name, (unsigned long)reg_info.size,

                            (unsigned long)reg_info.offset,

                            (unsigned long)reg_info.flags);



    vdev->rom_size = size = reg_info.size;

    vdev->rom_offset = reg_info.offset;



    if (!vdev->rom_size) {

        vdev->rom_read_failed = true;

        error_report("vfio-pci: Cannot read device rom at "

                    "%s", vdev->vbasedev.name);

        error_printf("Device option ROM contents are probably invalid "

                    "(check dmesg).\nSkip option ROM probe with rombar=0, "

                    "or load from file with romfile=\n");

        return;

    }



    vdev->rom = g_malloc(size);

    memset(vdev->rom, 0xff, size);



    while (size) {

        bytes = pread(vdev->vbasedev.fd, vdev->rom + off,

                      size, vdev->rom_offset + off);

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
