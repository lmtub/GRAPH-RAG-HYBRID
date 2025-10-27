static void aux_slave_dev_print(Monitor *mon, DeviceState *dev, int indent)

{

    AUXBus *bus = AUX_BUS(qdev_get_parent_bus(dev));

    AUXSlave *s;



    /* Don't print anything if the device is I2C "bridge". */

    if (aux_bus_is_bridge(bus, dev)) {

        return;

    }



    s = AUX_SLAVE(dev);



    monitor_printf(mon, "%*smemory " TARGET_FMT_plx "/" TARGET_FMT_plx "\n",

                   indent, "",

                   object_property_get_int(OBJECT(s->mmio), "addr", NULL),

                   memory_region_size(s->mmio));

}
