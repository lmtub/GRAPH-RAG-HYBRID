void vfio_pci_write_config(PCIDevice *pdev,

                           uint32_t addr, uint32_t val, int len)

{

    VFIOPCIDevice *vdev = DO_UPCAST(VFIOPCIDevice, pdev, pdev);

    uint32_t val_le = cpu_to_le32(val);



    trace_vfio_pci_write_config(vdev->vbasedev.name, addr, val, len);



    /* Write everything to VFIO, let it filter out what we can't write */

    if (pwrite(vdev->vbasedev.fd, &val_le, len, vdev->config_offset + addr)

                != len) {

        error_report("%s(%04x:%02x:%02x.%x, 0x%x, 0x%x, 0x%x) failed: %m",

                     __func__, vdev->host.domain, vdev->host.bus,

                     vdev->host.slot, vdev->host.function, addr, val, len);

    }



    /* MSI/MSI-X Enabling/Disabling */

    if (pdev->cap_present & QEMU_PCI_CAP_MSI &&

        ranges_overlap(addr, len, pdev->msi_cap, vdev->msi_cap_size)) {

        int is_enabled, was_enabled = msi_enabled(pdev);



        pci_default_write_config(pdev, addr, val, len);



        is_enabled = msi_enabled(pdev);



        if (!was_enabled) {

            if (is_enabled) {

                vfio_msi_enable(vdev);

            }

        } else {

            if (!is_enabled) {

                vfio_msi_disable(vdev);

            } else {

                vfio_update_msi(vdev);

            }

        }

    } else if (pdev->cap_present & QEMU_PCI_CAP_MSIX &&

        ranges_overlap(addr, len, pdev->msix_cap, MSIX_CAP_LENGTH)) {

        int is_enabled, was_enabled = msix_enabled(pdev);



        pci_default_write_config(pdev, addr, val, len);



        is_enabled = msix_enabled(pdev);



        if (!was_enabled && is_enabled) {

            vfio_msix_enable(vdev);

        } else if (was_enabled && !is_enabled) {

            vfio_msix_disable(vdev);

        }

    } else {

        /* Write everything to QEMU to keep emulated bits correct */

        pci_default_write_config(pdev, addr, val, len);

    }

}
