static void kvmclock_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = kvmclock_realize;

    dc->vmsd = &kvmclock_vmsd;

    dc->props = kvmclock_properties;






}