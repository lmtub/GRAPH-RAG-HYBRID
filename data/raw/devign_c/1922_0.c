static void spapr_cpu_core_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    dc->realize = spapr_cpu_core_realize;

}
