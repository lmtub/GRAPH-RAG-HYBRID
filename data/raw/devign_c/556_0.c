type_init(pflash_cfi02_register_types)



pflash_t *pflash_cfi02_register(hwaddr base,

                                DeviceState *qdev, const char *name,

                                hwaddr size,

                                BlockDriverState *bs, uint32_t sector_len,

                                int nb_blocs, int nb_mappings, int width,

                                uint16_t id0, uint16_t id1,

                                uint16_t id2, uint16_t id3,

                                uint16_t unlock_addr0, uint16_t unlock_addr1,

                                int be)

{

    DeviceState *dev = qdev_create(NULL, TYPE_CFI_PFLASH02);



    if (bs && qdev_prop_set_drive(dev, "drive", bs)) {

        abort();

    }

    qdev_prop_set_uint32(dev, "num-blocks", nb_blocs);

    qdev_prop_set_uint32(dev, "sector-length", sector_len);

    qdev_prop_set_uint8(dev, "width", width);

    qdev_prop_set_uint8(dev, "mappings", nb_mappings);

    qdev_prop_set_uint8(dev, "big-endian", !!be);

    qdev_prop_set_uint16(dev, "id0", id0);

    qdev_prop_set_uint16(dev, "id1", id1);

    qdev_prop_set_uint16(dev, "id2", id2);

    qdev_prop_set_uint16(dev, "id3", id3);

    qdev_prop_set_uint16(dev, "unlock-addr0", unlock_addr0);

    qdev_prop_set_uint16(dev, "unlock-addr1", unlock_addr1);

    qdev_prop_set_string(dev, "name", name);

    qdev_init_nofail(dev);



    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, base);

    return CFI_PFLASH02(dev);

}
