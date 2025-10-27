ISADevice *isa_create_simple(const char *name)

{

    ISADevice *dev;



    dev = isa_create(name);

    if (qdev_init(&dev->qdev) != 0) {

        return NULL;

    }

    return dev;

}
