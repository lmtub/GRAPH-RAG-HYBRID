static int hda_codec_dev_exit(DeviceState *qdev)

{

    HDACodecDevice *dev = HDA_CODEC_DEVICE(qdev);

    HDACodecDeviceClass *cdc = HDA_CODEC_DEVICE_GET_CLASS(dev);



    if (cdc->exit) {

        cdc->exit(dev);

    }

    return 0;

}
