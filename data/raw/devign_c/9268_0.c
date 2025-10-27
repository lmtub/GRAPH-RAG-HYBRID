static uint16_t shpc_get_status(SHPCDevice *shpc, int slot, uint16_t msk)

{

    uint8_t *status = shpc->config + SHPC_SLOT_STATUS(slot);

    return (pci_get_word(status) & msk) >> (ffs(msk) - 1);

}
