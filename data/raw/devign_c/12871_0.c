static void pc_fw_cfg_guest_info(PcGuestInfo *guest_info)

{

    PcRomPciInfo *info;

    if (!guest_info->has_pci_info) {

        return;

    }



    info = g_malloc(sizeof *info);

    info->w32_min = cpu_to_le64(guest_info->pci_info.w32.begin);

    info->w32_max = cpu_to_le64(guest_info->pci_info.w32.end);

    info->w64_min = cpu_to_le64(guest_info->pci_info.w64.begin);

    info->w64_max = cpu_to_le64(guest_info->pci_info.w64.end);

    /* Pass PCI hole info to guest via a side channel.

     * Required so guest PCI enumeration does the right thing. */

    fw_cfg_add_file(guest_info->fw_cfg, "etc/pci-info", info, sizeof *info);

}
