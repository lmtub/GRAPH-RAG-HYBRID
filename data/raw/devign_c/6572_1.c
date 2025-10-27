void qdev_init_gpio_out_named(DeviceState *dev, qemu_irq *pins,
                              const char *name, int n)
{
    NamedGPIOList *gpio_list = qdev_get_named_gpio_list(dev, name);
    assert(gpio_list->num_in == 0 || !name);
    assert(gpio_list->num_out == 0);
    gpio_list->num_out = n;
    gpio_list->out = pins;