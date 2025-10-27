static void shpc_set_status(SHPCDevice *shpc,

                            int slot, uint8_t value, uint16_t msk)

{

    uint8_t *status = shpc->config + SHPC_SLOT_STATUS(slot);

    pci_word_test_and_clear_mask(status, msk);

    pci_word_test_and_set_mask(status, value << (ffs(msk) - 1));

}
