static void spapr_create_nvram(sPAPREnvironment *spapr)

{

    DeviceState *dev = qdev_create(&spapr->vio_bus->bus, "spapr-nvram");

    DriveInfo *dinfo = drive_get(IF_PFLASH, 0, 0);



    if (dinfo) {

        qdev_prop_set_drive_nofail(dev, "drive",

                                   blk_bs(blk_by_legacy_dinfo(dinfo)));

    }



    qdev_init_nofail(dev);



    spapr->nvram = (struct sPAPRNVRAM *)dev;

}
