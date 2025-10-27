static void dp8393x_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    set_bit(DEVICE_CATEGORY_NETWORK, dc->categories);

    dc->realize = dp8393x_realize;

    dc->reset = dp8393x_reset;

    dc->vmsd = &vmstate_dp8393x;

    dc->props = dp8393x_properties;



}