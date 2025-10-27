static void spitz_i2c_setup(PXA2xxState *cpu)

{

    /* Attach the CPU on one end of our I2C bus.  */

    i2c_bus *bus = pxa2xx_i2c_bus(cpu->i2c[0]);



#ifdef HAS_AUDIO

    DeviceState *wm;



    /* Attach a WM8750 to the bus */

    wm = i2c_create_slave(bus, "wm8750", 0);



    spitz_wm8750_addr(wm, 0, 0);

    pxa2xx_gpio_out_set(cpu->gpio, SPITZ_GPIO_WM,

                    qemu_allocate_irqs(spitz_wm8750_addr, wm, 1)[0]);

    /* .. and to the sound interface.  */

    cpu->i2s->opaque = wm;

    cpu->i2s->codec_out = wm8750_dac_dat;

    cpu->i2s->codec_in = wm8750_adc_dat;

    wm8750_data_req_set(wm, cpu->i2s->data_req, cpu->i2s);

#endif

}
