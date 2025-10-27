static int twl92230_init(i2c_slave *i2c)

{

    MenelausState *s = FROM_I2C_SLAVE(MenelausState, i2c);



    s->rtc.hz_tm = qemu_new_timer(rt_clock, menelaus_rtc_hz, s);

    /* Three output pins plus one interrupt pin.  */

    qdev_init_gpio_out(&i2c->qdev, s->out, 4);

    qdev_init_gpio_in(&i2c->qdev, menelaus_gpio_set, 3);

    s->pwrbtn = qemu_allocate_irqs(menelaus_pwrbtn_set, s, 1)[0];



    menelaus_reset(&s->i2c);



    return 0;

}
