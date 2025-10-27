static int usb_qdev_init(DeviceState *qdev, DeviceInfo *base)

{

    USBDevice *dev = DO_UPCAST(USBDevice, qdev, qdev);

    USBDeviceInfo *info = DO_UPCAST(USBDeviceInfo, qdev, base);

    int rc;



    pstrcpy(dev->product_desc, sizeof(dev->product_desc), info->product_desc);

    dev->info = info;

    dev->auto_attach = 1;

    QLIST_INIT(&dev->strings);

    rc = dev->info->init(dev);

    if (rc == 0 && dev->auto_attach)

        rc = usb_device_attach(dev);

    return rc;

}
