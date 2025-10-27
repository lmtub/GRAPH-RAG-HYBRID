pflash_t *pflash_cfi01_register(target_phys_addr_t base,

                                DeviceState *qdev, const char *name,

                                target_phys_addr_t size,

                                BlockDriverState *bs, uint32_t sector_len,

                                int nb_blocs, int width,

                                uint16_t id0, uint16_t id1,

                                uint16_t id2, uint16_t id3, int be)

{

    pflash_t *pfl;

    target_phys_addr_t total_len;

    int ret;



    total_len = sector_len * nb_blocs;



    /* XXX: to be fixed */

#if 0

    if (total_len != (8 * 1024 * 1024) && total_len != (16 * 1024 * 1024) &&

        total_len != (32 * 1024 * 1024) && total_len != (64 * 1024 * 1024))

        return NULL;

#endif



    pfl = g_malloc0(sizeof(pflash_t));



    memory_region_init_rom_device(

        &pfl->mem, be ? &pflash_cfi01_ops_be : &pflash_cfi01_ops_le, pfl,

        name, size);

    vmstate_register_ram(&pfl->mem, qdev);

    pfl->storage = memory_region_get_ram_ptr(&pfl->mem);

    memory_region_add_subregion(get_system_memory(), base, &pfl->mem);



    pfl->bs = bs;

    if (pfl->bs) {

        /* read the initial flash content */

        ret = bdrv_read(pfl->bs, 0, pfl->storage, total_len >> 9);

        if (ret < 0) {

            memory_region_del_subregion(get_system_memory(), &pfl->mem);

            vmstate_unregister_ram(&pfl->mem, qdev);

            memory_region_destroy(&pfl->mem);

            g_free(pfl);

            return NULL;

        }

        bdrv_attach_dev_nofail(pfl->bs, pfl);

    }



    if (pfl->bs) {

        pfl->ro = bdrv_is_read_only(pfl->bs);

    } else {

        pfl->ro = 0;

    }



    pfl->timer = qemu_new_timer_ns(vm_clock, pflash_timer, pfl);

    pfl->base = base;

    pfl->sector_len = sector_len;

    pfl->total_len = total_len;

    pfl->width = width;

    pfl->wcycle = 0;

    pfl->cmd = 0;

    pfl->status = 0;

    pfl->ident[0] = id0;

    pfl->ident[1] = id1;

    pfl->ident[2] = id2;

    pfl->ident[3] = id3;

    /* Hardcoded CFI table */

    pfl->cfi_len = 0x52;

    /* Standard "QRY" string */

    pfl->cfi_table[0x10] = 'Q';

    pfl->cfi_table[0x11] = 'R';

    pfl->cfi_table[0x12] = 'Y';

    /* Command set (Intel) */

    pfl->cfi_table[0x13] = 0x01;

    pfl->cfi_table[0x14] = 0x00;

    /* Primary extended table address (none) */

    pfl->cfi_table[0x15] = 0x31;

    pfl->cfi_table[0x16] = 0x00;

    /* Alternate command set (none) */

    pfl->cfi_table[0x17] = 0x00;

    pfl->cfi_table[0x18] = 0x00;

    /* Alternate extended table (none) */

    pfl->cfi_table[0x19] = 0x00;

    pfl->cfi_table[0x1A] = 0x00;

    /* Vcc min */

    pfl->cfi_table[0x1B] = 0x45;

    /* Vcc max */

    pfl->cfi_table[0x1C] = 0x55;

    /* Vpp min (no Vpp pin) */

    pfl->cfi_table[0x1D] = 0x00;

    /* Vpp max (no Vpp pin) */

    pfl->cfi_table[0x1E] = 0x00;

    /* Reserved */

    pfl->cfi_table[0x1F] = 0x07;

    /* Timeout for min size buffer write */

    pfl->cfi_table[0x20] = 0x07;

    /* Typical timeout for block erase */

    pfl->cfi_table[0x21] = 0x0a;

    /* Typical timeout for full chip erase (4096 ms) */

    pfl->cfi_table[0x22] = 0x00;

    /* Reserved */

    pfl->cfi_table[0x23] = 0x04;

    /* Max timeout for buffer write */

    pfl->cfi_table[0x24] = 0x04;

    /* Max timeout for block erase */

    pfl->cfi_table[0x25] = 0x04;

    /* Max timeout for chip erase */

    pfl->cfi_table[0x26] = 0x00;

    /* Device size */

    pfl->cfi_table[0x27] = ctz32(total_len); // + 1;

    /* Flash device interface (8 & 16 bits) */

    pfl->cfi_table[0x28] = 0x02;

    pfl->cfi_table[0x29] = 0x00;

    /* Max number of bytes in multi-bytes write */

    if (width == 1) {

        pfl->cfi_table[0x2A] = 0x08;

    } else {

        pfl->cfi_table[0x2A] = 0x0B;

    }

    pfl->writeblock_size = 1 << pfl->cfi_table[0x2A];



    pfl->cfi_table[0x2B] = 0x00;

    /* Number of erase block regions (uniform) */

    pfl->cfi_table[0x2C] = 0x01;

    /* Erase block region 1 */

    pfl->cfi_table[0x2D] = nb_blocs - 1;

    pfl->cfi_table[0x2E] = (nb_blocs - 1) >> 8;

    pfl->cfi_table[0x2F] = sector_len >> 8;

    pfl->cfi_table[0x30] = sector_len >> 16;



    /* Extended */

    pfl->cfi_table[0x31] = 'P';

    pfl->cfi_table[0x32] = 'R';

    pfl->cfi_table[0x33] = 'I';



    pfl->cfi_table[0x34] = '1';

    pfl->cfi_table[0x35] = '1';



    pfl->cfi_table[0x36] = 0x00;

    pfl->cfi_table[0x37] = 0x00;

    pfl->cfi_table[0x38] = 0x00;

    pfl->cfi_table[0x39] = 0x00;



    pfl->cfi_table[0x3a] = 0x00;



    pfl->cfi_table[0x3b] = 0x00;

    pfl->cfi_table[0x3c] = 0x00;



    return pfl;

}
