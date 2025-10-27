ISABus *isa_bus_new(DeviceState *dev)

{

    if (isabus) {

        fprintf(stderr, "Can't create a second ISA bus\n");

        return NULL;

    }

    if (NULL == dev) {

        dev = qdev_create(NULL, "isabus-bridge");

        qdev_init(dev);

    }



    isabus = FROM_QBUS(ISABus, qbus_create(&isa_bus_info, dev, NULL));

    return isabus;

}
