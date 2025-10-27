int msi_init(struct PCIDevice *dev, uint8_t offset,

             unsigned int nr_vectors, bool msi64bit, bool msi_per_vector_mask)

{

    unsigned int vectors_order;

    uint16_t flags;

    uint8_t cap_size;

    int config_offset;



    if (!msi_supported) {

        return -ENOTSUP;

    }



    MSI_DEV_PRINTF(dev,

                   "init offset: 0x%"PRIx8" vector: %"PRId8

                   " 64bit %d mask %d\n",

                   offset, nr_vectors, msi64bit, msi_per_vector_mask);



    assert(!(nr_vectors & (nr_vectors - 1)));   /* power of 2 */

    assert(nr_vectors > 0);

    assert(nr_vectors <= PCI_MSI_VECTORS_MAX);

    /* the nr of MSI vectors is up to 32 */

    vectors_order = ffs(nr_vectors) - 1;



    flags = vectors_order << (ffs(PCI_MSI_FLAGS_QMASK) - 1);

    if (msi64bit) {

        flags |= PCI_MSI_FLAGS_64BIT;

    }

    if (msi_per_vector_mask) {

        flags |= PCI_MSI_FLAGS_MASKBIT;

    }



    cap_size = msi_cap_sizeof(flags);

    config_offset = pci_add_capability(dev, PCI_CAP_ID_MSI, offset, cap_size);

    if (config_offset < 0) {

        return config_offset;

    }



    dev->msi_cap = config_offset;

    dev->cap_present |= QEMU_PCI_CAP_MSI;



    pci_set_word(dev->config + msi_flags_off(dev), flags);

    pci_set_word(dev->wmask + msi_flags_off(dev),

                 PCI_MSI_FLAGS_QSIZE | PCI_MSI_FLAGS_ENABLE);

    pci_set_long(dev->wmask + msi_address_lo_off(dev),

                 PCI_MSI_ADDRESS_LO_MASK);

    if (msi64bit) {

        pci_set_long(dev->wmask + msi_address_hi_off(dev), 0xffffffff);

    }

    pci_set_word(dev->wmask + msi_data_off(dev, msi64bit), 0xffff);



    if (msi_per_vector_mask) {

        /* Make mask bits 0 to nr_vectors - 1 writable. */

        pci_set_long(dev->wmask + msi_mask_off(dev, msi64bit),

                     0xffffffff >> (PCI_MSI_VECTORS_MAX - nr_vectors));

    }

    return config_offset;

}
