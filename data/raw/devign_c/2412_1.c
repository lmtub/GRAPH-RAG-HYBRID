static void pflash_cfi02_realize(DeviceState *dev, Error **errp)
{
    pflash_t *pfl = CFI_PFLASH02(dev);
    uint32_t chip_len;
    int ret;
    Error *local_err = NULL;
    chip_len = pfl->sector_len * pfl->nb_blocs;
    /* XXX: to be fixed */
#if 0
    if (total_len != (8 * 1024 * 1024) && total_len != (16 * 1024 * 1024) &&
        total_len != (32 * 1024 * 1024) && total_len != (64 * 1024 * 1024))
        return NULL;
#endif
    memory_region_init_rom_device(&pfl->orig_mem, OBJECT(pfl), pfl->be ?
                                  &pflash_cfi02_ops_be : &pflash_cfi02_ops_le,
                                  pfl, pfl->name, chip_len, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
    vmstate_register_ram(&pfl->orig_mem, DEVICE(pfl));
    pfl->storage = memory_region_get_ram_ptr(&pfl->orig_mem);
    pfl->chip_len = chip_len;
    if (pfl->blk) {
        /* read the initial flash content */
        ret = blk_pread(pfl->blk, 0, pfl->storage, chip_len);
        if (ret < 0) {
            vmstate_unregister_ram(&pfl->orig_mem, DEVICE(pfl));
            error_setg(errp, "failed to read the initial flash content");
    pflash_setup_mappings(pfl);
    pfl->rom_mode = 1;
    sysbus_init_mmio(SYS_BUS_DEVICE(dev), &pfl->mem);
    if (pfl->blk) {
        pfl->ro = blk_is_read_only(pfl->blk);
    } else {
        pfl->ro = 0;
    pfl->timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, pflash_timer, pfl);
    pfl->wcycle = 0;
    pfl->cmd = 0;
    pfl->status = 0;
    /* Hardcoded CFI table (mostly from SG29 Spansion flash) */
    pfl->cfi_len = 0x52;
    /* Standard "QRY" string */
    pfl->cfi_table[0x10] = 'Q';
    pfl->cfi_table[0x11] = 'R';
    pfl->cfi_table[0x12] = 'Y';
    /* Command set (AMD/Fujitsu) */
    pfl->cfi_table[0x13] = 0x02;
    pfl->cfi_table[0x14] = 0x00;
    /* Primary extended table address */
    pfl->cfi_table[0x15] = 0x31;
    pfl->cfi_table[0x16] = 0x00;
    /* Alternate command set (none) */
    pfl->cfi_table[0x17] = 0x00;
    pfl->cfi_table[0x18] = 0x00;
    /* Alternate extended table (none) */
    pfl->cfi_table[0x19] = 0x00;
    pfl->cfi_table[0x1A] = 0x00;
    /* Vcc min */
    pfl->cfi_table[0x1B] = 0x27;
    /* Vcc max */
    pfl->cfi_table[0x1C] = 0x36;
    /* Vpp min (no Vpp pin) */
    pfl->cfi_table[0x1D] = 0x00;
    /* Vpp max (no Vpp pin) */
    pfl->cfi_table[0x1E] = 0x00;
    /* Reserved */
    pfl->cfi_table[0x1F] = 0x07;
    /* Timeout for min size buffer write (NA) */
    pfl->cfi_table[0x20] = 0x00;
    /* Typical timeout for block erase (512 ms) */
    pfl->cfi_table[0x21] = 0x09;
    /* Typical timeout for full chip erase (4096 ms) */
    pfl->cfi_table[0x22] = 0x0C;
    /* Reserved */
    pfl->cfi_table[0x23] = 0x01;
    /* Max timeout for buffer write (NA) */
    pfl->cfi_table[0x24] = 0x00;
    /* Max timeout for block erase */
    pfl->cfi_table[0x25] = 0x0A;
    /* Max timeout for chip erase */
    pfl->cfi_table[0x26] = 0x0D;
    /* Device size */
    pfl->cfi_table[0x27] = ctz32(chip_len);
    /* Flash device interface (8 & 16 bits) */
    pfl->cfi_table[0x28] = 0x02;
    pfl->cfi_table[0x29] = 0x00;
    /* Max number of bytes in multi-bytes write */
    /* XXX: disable buffered write as it's not supported */
    //    pfl->cfi_table[0x2A] = 0x05;
    pfl->cfi_table[0x2A] = 0x00;
    pfl->cfi_table[0x2B] = 0x00;
    /* Number of erase block regions (uniform) */
    pfl->cfi_table[0x2C] = 0x01;
    /* Erase block region 1 */
    pfl->cfi_table[0x2D] = pfl->nb_blocs - 1;
    pfl->cfi_table[0x2E] = (pfl->nb_blocs - 1) >> 8;
    pfl->cfi_table[0x2F] = pfl->sector_len >> 8;
    pfl->cfi_table[0x30] = pfl->sector_len >> 16;
    /* Extended */
    pfl->cfi_table[0x31] = 'P';
    pfl->cfi_table[0x32] = 'R';
    pfl->cfi_table[0x33] = 'I';
    pfl->cfi_table[0x34] = '1';
    pfl->cfi_table[0x35] = '0';
    pfl->cfi_table[0x36] = 0x00;
    pfl->cfi_table[0x37] = 0x00;
    pfl->cfi_table[0x38] = 0x00;
    pfl->cfi_table[0x39] = 0x00;
    pfl->cfi_table[0x3a] = 0x00;
    pfl->cfi_table[0x3b] = 0x00;
    pfl->cfi_table[0x3c] = 0x00;