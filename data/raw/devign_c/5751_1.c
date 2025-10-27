static void interface_set_client_capabilities(QXLInstance *sin,
                                              uint8_t client_present,
                                              uint8_t caps[58])
{
    PCIQXLDevice *qxl = container_of(sin, PCIQXLDevice, ssd.qxl);
    qxl->shadow_rom.client_present = client_present;
    memcpy(qxl->shadow_rom.client_capabilities, caps, sizeof(caps));
    qxl->rom->client_present = client_present;
    memcpy(qxl->rom->client_capabilities, caps, sizeof(caps));
    qxl_rom_set_dirty(qxl);
    qxl_send_events(qxl, QXL_INTERRUPT_CLIENT);