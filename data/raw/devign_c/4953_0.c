static void create_one_flash(const char *name, hwaddr flashbase,

                             hwaddr flashsize)

{

    /* Create and map a single flash device. We use the same

     * parameters as the flash devices on the Versatile Express board.

     */

    DriveInfo *dinfo = drive_get_next(IF_PFLASH);

    DeviceState *dev = qdev_create(NULL, "cfi.pflash01");

    const uint64_t sectorlength = 256 * 1024;



    if (dinfo && qdev_prop_set_drive(dev, "drive",

                                     blk_bs(blk_by_legacy_dinfo(dinfo)))) {

        abort();

    }



    qdev_prop_set_uint32(dev, "num-blocks", flashsize / sectorlength);

    qdev_prop_set_uint64(dev, "sector-length", sectorlength);

    qdev_prop_set_uint8(dev, "width", 4);

    qdev_prop_set_uint8(dev, "device-width", 2);

    qdev_prop_set_uint8(dev, "big-endian", 0);

    qdev_prop_set_uint16(dev, "id0", 0x89);

    qdev_prop_set_uint16(dev, "id1", 0x18);

    qdev_prop_set_uint16(dev, "id2", 0x00);

    qdev_prop_set_uint16(dev, "id3", 0x00);

    qdev_prop_set_string(dev, "name", name);

    qdev_init_nofail(dev);



    sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0, flashbase);

}
