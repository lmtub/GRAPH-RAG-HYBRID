void msi_notify(PCIDevice *dev, unsigned int vector)

{

    uint16_t flags = pci_get_word(dev->config + msi_flags_off(dev));

    bool msi64bit = flags & PCI_MSI_FLAGS_64BIT;

    unsigned int nr_vectors = msi_nr_vectors(flags);

    MSIMessage msg;



    assert(vector < nr_vectors);

    if (msi_is_masked(dev, vector)) {

        assert(flags & PCI_MSI_FLAGS_MASKBIT);

        pci_long_test_and_set_mask(

            dev->config + msi_pending_off(dev, msi64bit), 1U << vector);

        MSI_DEV_PRINTF(dev, "pending vector 0x%x\n", vector);

        return;

    }



    msg = msi_get_message(dev, vector);



    MSI_DEV_PRINTF(dev,

                   "notify vector 0x%x"

                   " address: 0x%"PRIx64" data: 0x%"PRIx32"\n",

                   vector, msg.address, msg.data);

    stl_le_phys(&address_space_memory, msg.address, msg.data);

}
