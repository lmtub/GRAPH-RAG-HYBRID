static void stellaris_init(const char *kernel_filename, const char *cpu_model,

                           stellaris_board_info *board)

{

    static const int uart_irq[] = {5, 6, 33, 34};

    static const int timer_irq[] = {19, 21, 23, 35};

    static const uint32_t gpio_addr[7] =

      { 0x40004000, 0x40005000, 0x40006000, 0x40007000,

        0x40024000, 0x40025000, 0x40026000};

    static const int gpio_irq[7] = {0, 1, 2, 3, 4, 30, 31};



    qemu_irq *pic;

    DeviceState *gpio_dev[7];

    qemu_irq gpio_in[7][8];

    qemu_irq gpio_out[7][8];

    qemu_irq adc;

    int sram_size;

    int flash_size;

    I2CBus *i2c;

    DeviceState *dev;

    int i;

    int j;



    MemoryRegion *sram = g_new(MemoryRegion, 1);

    MemoryRegion *flash = g_new(MemoryRegion, 1);

    MemoryRegion *system_memory = get_system_memory();



    flash_size = (((board->dc0 & 0xffff) + 1) << 1) * 1024;

    sram_size = ((board->dc0 >> 18) + 1) * 1024;



    /* Flash programming is done via the SCU, so pretend it is ROM.  */

    memory_region_init_ram(flash, NULL, "stellaris.flash", flash_size,

                           &error_abort);

    vmstate_register_ram_global(flash);

    memory_region_set_readonly(flash, true);

    memory_region_add_subregion(system_memory, 0, flash);



    memory_region_init_ram(sram, NULL, "stellaris.sram", sram_size,

                           &error_abort);

    vmstate_register_ram_global(sram);

    memory_region_add_subregion(system_memory, 0x20000000, sram);



    pic = armv7m_init(system_memory, flash_size, NUM_IRQ_LINES,

                      kernel_filename, cpu_model);



    if (board->dc1 & (1 << 16)) {

        dev = sysbus_create_varargs(TYPE_STELLARIS_ADC, 0x40038000,

                                    pic[14], pic[15], pic[16], pic[17], NULL);

        adc = qdev_get_gpio_in(dev, 0);

    } else {

        adc = NULL;

    }

    for (i = 0; i < 4; i++) {

        if (board->dc2 & (0x10000 << i)) {

            dev = sysbus_create_simple(TYPE_STELLARIS_GPTM,

                                       0x40030000 + i * 0x1000,

                                       pic[timer_irq[i]]);

            /* TODO: This is incorrect, but we get away with it because

               the ADC output is only ever pulsed.  */

            qdev_connect_gpio_out(dev, 0, adc);

        }

    }



    stellaris_sys_init(0x400fe000, pic[28], board, nd_table[0].macaddr.a);



    for (i = 0; i < 7; i++) {

        if (board->dc4 & (1 << i)) {

            gpio_dev[i] = sysbus_create_simple("pl061_luminary", gpio_addr[i],

                                               pic[gpio_irq[i]]);

            for (j = 0; j < 8; j++) {

                gpio_in[i][j] = qdev_get_gpio_in(gpio_dev[i], j);

                gpio_out[i][j] = NULL;

            }

        }

    }



    if (board->dc2 & (1 << 12)) {

        dev = sysbus_create_simple(TYPE_STELLARIS_I2C, 0x40020000, pic[8]);

        i2c = (I2CBus *)qdev_get_child_bus(dev, "i2c");

        if (board->peripherals & BP_OLED_I2C) {

            i2c_create_slave(i2c, "ssd0303", 0x3d);

        }

    }



    for (i = 0; i < 4; i++) {

        if (board->dc2 & (1 << i)) {

            sysbus_create_simple("pl011_luminary", 0x4000c000 + i * 0x1000,

                                 pic[uart_irq[i]]);

        }

    }

    if (board->dc2 & (1 << 4)) {

        dev = sysbus_create_simple("pl022", 0x40008000, pic[7]);

        if (board->peripherals & BP_OLED_SSI) {

            void *bus;

            DeviceState *sddev;

            DeviceState *ssddev;



            /* Some boards have both an OLED controller and SD card connected to

             * the same SSI port, with the SD card chip select connected to a

             * GPIO pin.  Technically the OLED chip select is connected to the

             * SSI Fss pin.  We do not bother emulating that as both devices

             * should never be selected simultaneously, and our OLED controller

             * ignores stray 0xff commands that occur when deselecting the SD

             * card.

             */

            bus = qdev_get_child_bus(dev, "ssi");



            sddev = ssi_create_slave(bus, "ssi-sd");

            ssddev = ssi_create_slave(bus, "ssd0323");

            gpio_out[GPIO_D][0] = qemu_irq_split(

                    qdev_get_gpio_in_named(sddev, SSI_GPIO_CS, 0),

                    qdev_get_gpio_in_named(ssddev, SSI_GPIO_CS, 0));

            gpio_out[GPIO_C][7] = qdev_get_gpio_in(ssddev, 0);



            /* Make sure the select pin is high.  */

            qemu_irq_raise(gpio_out[GPIO_D][0]);

        }

    }

    if (board->dc4 & (1 << 28)) {

        DeviceState *enet;



        qemu_check_nic_model(&nd_table[0], "stellaris");



        enet = qdev_create(NULL, "stellaris_enet");

        qdev_set_nic_properties(enet, &nd_table[0]);

        qdev_init_nofail(enet);

        sysbus_mmio_map(SYS_BUS_DEVICE(enet), 0, 0x40048000);

        sysbus_connect_irq(SYS_BUS_DEVICE(enet), 0, pic[42]);

    }

    if (board->peripherals & BP_GAMEPAD) {

        qemu_irq gpad_irq[5];

        static const int gpad_keycode[5] = { 0xc8, 0xd0, 0xcb, 0xcd, 0x1d };



        gpad_irq[0] = qemu_irq_invert(gpio_in[GPIO_E][0]); /* up */

        gpad_irq[1] = qemu_irq_invert(gpio_in[GPIO_E][1]); /* down */

        gpad_irq[2] = qemu_irq_invert(gpio_in[GPIO_E][2]); /* left */

        gpad_irq[3] = qemu_irq_invert(gpio_in[GPIO_E][3]); /* right */

        gpad_irq[4] = qemu_irq_invert(gpio_in[GPIO_F][1]); /* select */



        stellaris_gamepad_init(5, gpad_irq, gpad_keycode);

    }

    for (i = 0; i < 7; i++) {

        if (board->dc4 & (1 << i)) {

            for (j = 0; j < 8; j++) {

                if (gpio_out[i][j]) {

                    qdev_connect_gpio_out(gpio_dev[i], j, gpio_out[i][j]);

                }

            }

        }

    }

}
