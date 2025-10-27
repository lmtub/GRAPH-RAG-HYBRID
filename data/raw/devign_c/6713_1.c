static bool qvirtio_pci_get_config_isr_status(QVirtioDevice *d)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    uint32_t data;



    if (dev->pdev->msix_enabled) {

        g_assert_cmpint(dev->config_msix_entry, !=, -1);

        if (qpci_msix_masked(dev->pdev, dev->config_msix_entry)) {

            /* No ISR checking should be done if masked, but read anyway */

            return qpci_msix_pending(dev->pdev, dev->config_msix_entry);

        } else {

            data = readl(dev->config_msix_addr);

            writel(dev->config_msix_addr, 0);

            return data == dev->config_msix_data;

        }

    } else {

        return qpci_io_readb(dev->pdev, dev->addr + QVIRTIO_PCI_ISR_STATUS) & 2;

    }

}
