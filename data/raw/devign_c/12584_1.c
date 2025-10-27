ISADevice *isa_try_create(ISABus *bus, const char *name)

{

    DeviceState *dev;



    if (!bus) {

        hw_error("Tried to create isa device %s with no isa bus present.",

                 name);

    }

    dev = qdev_try_create(BUS(bus), name);

    return ISA_DEVICE(dev);

}
