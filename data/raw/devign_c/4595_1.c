static uint64_t vfio_rtl8168_window_quirk_read(void *opaque,

                                               hwaddr addr, unsigned size)

{

    VFIOQuirk *quirk = opaque;

    VFIOPCIDevice *vdev = quirk->vdev;



    switch (addr) {

    case 4: /* address */

        if (quirk->data.flags) {

            trace_vfio_rtl8168_window_quirk_read_fake(

                    memory_region_name(&quirk->mem),

                    vdev->vbasedev.name);



            return quirk->data.address_match ^ 0x80000000U;

        }

        break;

    case 0: /* data */

        if (quirk->data.flags) {

            uint64_t val;



            trace_vfio_rtl8168_window_quirk_read_table(

                    memory_region_name(&quirk->mem),

                    vdev->vbasedev.name);



            if (!(vdev->pdev.cap_present & QEMU_PCI_CAP_MSIX)) {

                return 0;

            }



            memory_region_dispatch_read(&vdev->pdev.msix_table_mmio,

                                        (hwaddr)(quirk->data.address_match

                                                 & 0xfff),

                                        &val,

                                        size,

                                        MEMTXATTRS_UNSPECIFIED);

            return val;

        }

    }



    trace_vfio_rtl8168_window_quirk_read_direct(memory_region_name(&quirk->mem),

                                                vdev->vbasedev.name);



    return vfio_region_read(&vdev->bars[quirk->data.bar].region,

                            addr + 0x70, size);

}
