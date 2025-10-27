void msix_write_config(PCIDevice *dev, uint32_t addr,

                       uint32_t val, int len)

{

    unsigned enable_pos = dev->msix_cap + MSIX_CONTROL_OFFSET;

    if (addr + len <= enable_pos || addr > enable_pos)

        return;



    if (msix_enabled(dev))

        qemu_set_irq(dev->irq[0], 0);

}
