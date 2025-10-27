static int vfio_initfn(PCIDevice *pdev)

{

    VFIOPCIDevice *vdev = DO_UPCAST(VFIOPCIDevice, pdev, pdev);

    VFIODevice *vbasedev_iter;

    VFIOGroup *group;

    char path[PATH_MAX], iommu_group_path[PATH_MAX], *group_name;

    ssize_t len;

    struct stat st;

    int groupid;

    int ret;



    /* Check that the host device exists */

    snprintf(path, sizeof(path),

             "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/",

             vdev->host.domain, vdev->host.bus, vdev->host.slot,

             vdev->host.function);

    if (stat(path, &st) < 0) {

        error_report("vfio: error: no such host device: %s", path);

        return -errno;

    }



    vdev->vbasedev.ops = &vfio_pci_ops;



    vdev->vbasedev.type = VFIO_DEVICE_TYPE_PCI;

    vdev->vbasedev.name = g_strdup_printf("%04x:%02x:%02x.%01x",

                                          vdev->host.domain, vdev->host.bus,

                                          vdev->host.slot, vdev->host.function);



    strncat(path, "iommu_group", sizeof(path) - strlen(path) - 1);



    len = readlink(path, iommu_group_path, sizeof(path));

    if (len <= 0 || len >= sizeof(path)) {

        error_report("vfio: error no iommu_group for device");

        return len < 0 ? -errno : -ENAMETOOLONG;

    }



    iommu_group_path[len] = 0;

    group_name = basename(iommu_group_path);



    if (sscanf(group_name, "%d", &groupid) != 1) {

        error_report("vfio: error reading %s: %m", path);

        return -errno;

    }



    trace_vfio_initfn(vdev->vbasedev.name, groupid);



    group = vfio_get_group(groupid, pci_device_iommu_address_space(pdev));

    if (!group) {

        error_report("vfio: failed to get group %d", groupid);

        return -ENOENT;

    }



    snprintf(path, sizeof(path), "%04x:%02x:%02x.%01x",

            vdev->host.domain, vdev->host.bus, vdev->host.slot,

            vdev->host.function);



    QLIST_FOREACH(vbasedev_iter, &group->device_list, next) {

        if (strcmp(vbasedev_iter->name, vdev->vbasedev.name) == 0) {

            error_report("vfio: error: device %s is already attached", path);

            vfio_put_group(group);

            return -EBUSY;

        }

    }



    ret = vfio_get_device(group, path, &vdev->vbasedev);

    if (ret) {

        error_report("vfio: failed to get device %s", path);

        vfio_put_group(group);

        return ret;

    }



    ret = vfio_populate_device(vdev);

    if (ret) {

        return ret;

    }



    /* Get a copy of config space */

    ret = pread(vdev->vbasedev.fd, vdev->pdev.config,

                MIN(pci_config_size(&vdev->pdev), vdev->config_size),

                vdev->config_offset);

    if (ret < (int)MIN(pci_config_size(&vdev->pdev), vdev->config_size)) {

        ret = ret < 0 ? -errno : -EFAULT;

        error_report("vfio: Failed to read device config space");

        return ret;

    }



    /* vfio emulates a lot for us, but some bits need extra love */

    vdev->emulated_config_bits = g_malloc0(vdev->config_size);



    /* QEMU can choose to expose the ROM or not */

    memset(vdev->emulated_config_bits + PCI_ROM_ADDRESS, 0xff, 4);



    /*

     * The PCI spec reserves vendor ID 0xffff as an invalid value.  The

     * device ID is managed by the vendor and need only be a 16-bit value.

     * Allow any 16-bit value for subsystem so they can be hidden or changed.

     */

    if (vdev->vendor_id != PCI_ANY_ID) {

        if (vdev->vendor_id >= 0xffff) {

            error_report("vfio: Invalid PCI vendor ID provided");

            return -EINVAL;

        }

        vfio_add_emulated_word(vdev, PCI_VENDOR_ID, vdev->vendor_id, ~0);

        trace_vfio_pci_emulated_vendor_id(vdev->vbasedev.name, vdev->vendor_id);

    } else {

        vdev->vendor_id = pci_get_word(pdev->config + PCI_VENDOR_ID);

    }



    if (vdev->device_id != PCI_ANY_ID) {

        if (vdev->device_id > 0xffff) {

            error_report("vfio: Invalid PCI device ID provided");

            return -EINVAL;

        }

        vfio_add_emulated_word(vdev, PCI_DEVICE_ID, vdev->device_id, ~0);

        trace_vfio_pci_emulated_device_id(vdev->vbasedev.name, vdev->device_id);

    } else {

        vdev->device_id = pci_get_word(pdev->config + PCI_DEVICE_ID);

    }



    if (vdev->sub_vendor_id != PCI_ANY_ID) {

        if (vdev->sub_vendor_id > 0xffff) {

            error_report("vfio: Invalid PCI subsystem vendor ID provided");

            return -EINVAL;

        }

        vfio_add_emulated_word(vdev, PCI_SUBSYSTEM_VENDOR_ID,

                               vdev->sub_vendor_id, ~0);

        trace_vfio_pci_emulated_sub_vendor_id(vdev->vbasedev.name,

                                              vdev->sub_vendor_id);

    }



    if (vdev->sub_device_id != PCI_ANY_ID) {

        if (vdev->sub_device_id > 0xffff) {

            error_report("vfio: Invalid PCI subsystem device ID provided");

            return -EINVAL;

        }

        vfio_add_emulated_word(vdev, PCI_SUBSYSTEM_ID, vdev->sub_device_id, ~0);

        trace_vfio_pci_emulated_sub_device_id(vdev->vbasedev.name,

                                              vdev->sub_device_id);

    }



    /* QEMU can change multi-function devices to single function, or reverse */

    vdev->emulated_config_bits[PCI_HEADER_TYPE] =

                                              PCI_HEADER_TYPE_MULTI_FUNCTION;



    /* Restore or clear multifunction, this is always controlled by QEMU */

    if (vdev->pdev.cap_present & QEMU_PCI_CAP_MULTIFUNCTION) {

        vdev->pdev.config[PCI_HEADER_TYPE] |= PCI_HEADER_TYPE_MULTI_FUNCTION;

    } else {

        vdev->pdev.config[PCI_HEADER_TYPE] &= ~PCI_HEADER_TYPE_MULTI_FUNCTION;

    }



    /*

     * Clear host resource mapping info.  If we choose not to register a

     * BAR, such as might be the case with the option ROM, we can get

     * confusing, unwritable, residual addresses from the host here.

     */

    memset(&vdev->pdev.config[PCI_BASE_ADDRESS_0], 0, 24);

    memset(&vdev->pdev.config[PCI_ROM_ADDRESS], 0, 4);



    vfio_pci_size_rom(vdev);



    ret = vfio_msix_early_setup(vdev);

    if (ret) {

        return ret;

    }



    vfio_map_bars(vdev);



    ret = vfio_add_capabilities(vdev);

    if (ret) {

        goto out_teardown;

    }



    /* QEMU emulates all of MSI & MSIX */

    if (pdev->cap_present & QEMU_PCI_CAP_MSIX) {

        memset(vdev->emulated_config_bits + pdev->msix_cap, 0xff,

               MSIX_CAP_LENGTH);

    }



    if (pdev->cap_present & QEMU_PCI_CAP_MSI) {

        memset(vdev->emulated_config_bits + pdev->msi_cap, 0xff,

               vdev->msi_cap_size);

    }



    if (vfio_pci_read_config(&vdev->pdev, PCI_INTERRUPT_PIN, 1)) {

        vdev->intx.mmap_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,

                                                  vfio_intx_mmap_enable, vdev);

        pci_device_set_intx_routing_notifier(&vdev->pdev, vfio_intx_update);

        ret = vfio_intx_enable(vdev);

        if (ret) {

            goto out_teardown;

        }

    }



    vfio_register_err_notifier(vdev);

    vfio_register_req_notifier(vdev);

    vfio_setup_resetfn_quirk(vdev);



    return 0;



out_teardown:

    pci_device_set_intx_routing_notifier(&vdev->pdev, NULL);

    vfio_teardown_msi(vdev);

    vfio_unregister_bars(vdev);

    return ret;

}
