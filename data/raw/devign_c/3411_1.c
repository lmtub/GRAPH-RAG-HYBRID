ISADevice *isa_create(const char *name)

{

    DeviceState *dev;



    if (!isabus) {

        fprintf(stderr, "Tried to create isa device %s with no isa bus present.\n", name);

        return NULL;

    }

    dev = qdev_create(&isabus->qbus, name);

    return DO_UPCAST(ISADevice, qdev, dev);

}
