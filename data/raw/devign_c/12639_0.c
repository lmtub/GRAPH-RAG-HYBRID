static int musicpal_lcd_init(SysBusDevice *dev)

{

    musicpal_lcd_state *s = FROM_SYSBUS(musicpal_lcd_state, dev);



    s->brightness = 7;



    memory_region_init_io(&s->iomem, &musicpal_lcd_ops, s,

                          "musicpal-lcd", MP_LCD_SIZE);

    sysbus_init_mmio(dev, &s->iomem);



    s->con = graphic_console_init(lcd_refresh, lcd_invalidate,

                                  NULL, NULL, s);

    qemu_console_resize(s->con, 128*3, 64*3);



    qdev_init_gpio_in(&dev->qdev, musicpal_lcd_gpio_brigthness_in, 3);



    return 0;

}
