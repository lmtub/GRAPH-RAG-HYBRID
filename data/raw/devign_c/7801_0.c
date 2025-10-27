i2c_bus *i2c_init_bus(DeviceState *parent, const char *name)

{

    i2c_bus *bus;



    bus = FROM_QBUS(i2c_bus, qbus_create(BUS_TYPE_I2C, sizeof(i2c_bus),

                                         parent, name));

    register_savevm("i2c_bus", -1, 1, i2c_bus_save, i2c_bus_load, bus);

    return bus;

}
