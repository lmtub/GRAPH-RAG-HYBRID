static int ide_drive_initfn(IDEDevice *dev)

{

    return ide_dev_initfn(dev,

                          bdrv_get_type_hint(dev->conf.bs) == BDRV_TYPE_CDROM

                          ? IDE_CD : IDE_HD);

}
