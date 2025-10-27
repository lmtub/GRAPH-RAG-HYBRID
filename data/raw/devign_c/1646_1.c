static void isabus_bridge_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);

    dc->fw_name = "isa";






}