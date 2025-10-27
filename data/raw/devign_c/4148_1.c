static bool qvirtio_pci_get_queue_isr_status(QVirtioDevice *d, QVirtQueue *vq)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    QVirtQueuePCI *vqpci = (QVirtQueuePCI *)vq;

    uint32_t data;



    if (dev->pdev->msix_enabled) {

        g_assert_cmpint(vqpci->msix_entry, !=, -1);

        if (qpci_msix_masked(dev->pdev, vqpci->msix_entry)) {

            /* No ISR checking should be done if masked, but read anyway */

            return qpci_msix_pending(dev->pdev, vqpci->msix_entry);

        } else {

            data = readl(vqpci->msix_addr);

            writel(vqpci->msix_addr, 0);

            return data == vqpci->msix_data;

        }

    } else {

        return qpci_io_readb(dev->pdev, dev->addr + QVIRTIO_PCI_ISR_STATUS) & 1;

    }

}
