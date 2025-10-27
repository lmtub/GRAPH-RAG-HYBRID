static void ds1338_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    I2CSlaveClass *k = I2C_SLAVE_CLASS(klass);



    k->init = ds1338_init;

    k->event = ds1338_event;

    k->recv = ds1338_recv;

    k->send = ds1338_send;

    dc->reset = ds1338_reset;

    dc->vmsd = &vmstate_ds1338;

}
