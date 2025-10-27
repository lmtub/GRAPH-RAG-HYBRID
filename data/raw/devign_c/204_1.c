static void pcie_cap_slot_hotplug_common(PCIDevice *hotplug_dev,

                                         DeviceState *dev,

                                         uint8_t **exp_cap, Error **errp)

{

    *exp_cap = hotplug_dev->config + hotplug_dev->exp.exp_cap;

    uint16_t sltsta = pci_get_word(*exp_cap + PCI_EXP_SLTSTA);



    PCIE_DEV_PRINTF(PCI_DEVICE(dev), "hotplug state: 0x%x\n", sltsta);

    if (sltsta & PCI_EXP_SLTSTA_EIS) {

        /* the slot is electromechanically locked.

         * This error is propagated up to qdev and then to HMP/QMP.

         */

        error_setg_errno(errp, -EBUSY, "slot is electromechanically locked");

    }

}
