static void aux_slave_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *k = DEVICE_CLASS(klass);



    set_bit(DEVICE_CATEGORY_MISC, k->categories);

    k->bus_type = TYPE_AUX_BUS;

}
