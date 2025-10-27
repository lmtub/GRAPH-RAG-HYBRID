static void spapr_rng_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);



    dc->realize = spapr_rng_realize;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

    dc->props = spapr_rng_properties;


}