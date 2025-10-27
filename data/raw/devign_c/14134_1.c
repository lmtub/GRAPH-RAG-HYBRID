static void event_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->bus_type = TYPE_SCLP_EVENTS_BUS;

    dc->unplug = qdev_simple_unplug_cb;

    dc->init = event_qdev_init;

    dc->exit = event_qdev_exit;

}
