static XICSState *try_create_xics(const char *type, int nr_servers,

                                  int nr_irqs)

{

    DeviceState *dev;



    dev = qdev_create(NULL, type);

    qdev_prop_set_uint32(dev, "nr_servers", nr_servers);

    qdev_prop_set_uint32(dev, "nr_irqs", nr_irqs);

    if (qdev_init(dev) < 0) {

        return NULL;

    }



    return XICS_COMMON(dev);

}
