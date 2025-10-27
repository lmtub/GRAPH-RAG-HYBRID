static int hda_codec_dev_init(DeviceState *qdev, DeviceInfo *base)

{

    HDACodecBus *bus = DO_UPCAST(HDACodecBus, qbus, qdev->parent_bus);

    HDACodecDevice *dev = DO_UPCAST(HDACodecDevice, qdev, qdev);

    HDACodecDeviceInfo *info = DO_UPCAST(HDACodecDeviceInfo, qdev, base);



    dev->info = info;

    if (dev->cad == -1) {

        dev->cad = bus->next_cad;

    }

    if (dev->cad > 15)

        return -1;

    bus->next_cad = dev->cad + 1;

    return info->init(dev);

}
