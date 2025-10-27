static void hda_codec_device_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *k = DEVICE_CLASS(klass);

    k->realize = hda_codec_dev_realize;

    k->exit = hda_codec_dev_exit;

    set_bit(DEVICE_CATEGORY_SOUND, k->categories);

    k->bus_type = TYPE_HDA_BUS;

    k->props = hda_props;

}
