static void icp_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->vmsd = &vmstate_icp_server;

    dc->realize = icp_realize;


}