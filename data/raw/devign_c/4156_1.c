static void e1000e_device_init(QPCIBus *bus, e1000e_device *d)

{

    uint32_t val;



    d->pci_dev = e1000e_device_find(bus);



    /* Enable the device */

    qpci_device_enable(d->pci_dev);



    /* Map BAR0 (mac registers) */

    d->mac_regs = qpci_iomap(d->pci_dev, 0, NULL);

    g_assert_nonnull(d->mac_regs);



    /* Reset the device */

    val = e1000e_macreg_read(d, E1000E_CTRL);

    e1000e_macreg_write(d, E1000E_CTRL, val | E1000E_CTRL_RESET);



    /* Enable and configure MSI-X */

    qpci_msix_enable(d->pci_dev);

    e1000e_macreg_write(d, E1000E_IVAR, E1000E_IVAR_TEST_CFG);



    /* Check the device status - link and speed */

    val = e1000e_macreg_read(d, E1000E_STATUS);

    g_assert_cmphex(val & (E1000E_STATUS_LU | E1000E_STATUS_ASDV1000),

        ==, E1000E_STATUS_LU | E1000E_STATUS_ASDV1000);



    /* Initialize TX/RX logic */

    e1000e_macreg_write(d, E1000E_RCTL, 0);

    e1000e_macreg_write(d, E1000E_TCTL, 0);



    /* Notify the device that the driver is ready */

    val = e1000e_macreg_read(d, E1000E_CTRL_EXT);

    e1000e_macreg_write(d, E1000E_CTRL_EXT,

        val | E1000E_CTRL_EXT_DRV_LOAD | E1000E_CTRL_EXT_TXLSFLOW);



    /* Allocate and setup TX ring */

    d->tx_ring = guest_alloc(test_alloc, E1000E_RING_LEN);

    g_assert(d->tx_ring != 0);



    e1000e_macreg_write(d, E1000E_TDBAL, (uint32_t) d->tx_ring);

    e1000e_macreg_write(d, E1000E_TDBAH, (uint32_t) (d->tx_ring >> 32));

    e1000e_macreg_write(d, E1000E_TDLEN, E1000E_RING_LEN);

    e1000e_macreg_write(d, E1000E_TDT, 0);

    e1000e_macreg_write(d, E1000E_TDH, 0);



    /* Enable transmit */

    e1000e_macreg_write(d, E1000E_TCTL, E1000E_TCTL_EN);



    /* Allocate and setup RX ring */

    d->rx_ring = guest_alloc(test_alloc, E1000E_RING_LEN);

    g_assert(d->rx_ring != 0);



    e1000e_macreg_write(d, E1000E_RDBAL, (uint32_t)d->rx_ring);

    e1000e_macreg_write(d, E1000E_RDBAH, (uint32_t)(d->rx_ring >> 32));

    e1000e_macreg_write(d, E1000E_RDLEN, E1000E_RING_LEN);

    e1000e_macreg_write(d, E1000E_RDT, 0);

    e1000e_macreg_write(d, E1000E_RDH, 0);



    /* Enable receive */

    e1000e_macreg_write(d, E1000E_RFCTL, E1000E_RFCTL_EXTEN);

    e1000e_macreg_write(d, E1000E_RCTL, E1000E_RCTL_EN  |

                                        E1000E_RCTL_UPE |

                                        E1000E_RCTL_MPE);



    /* Enable all interrupts */

    e1000e_macreg_write(d, E1000E_IMS, 0xFFFFFFFF);

}
