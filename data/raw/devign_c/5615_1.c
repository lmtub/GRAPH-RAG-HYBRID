static void fsl_imx25_realize(DeviceState *dev, Error **errp)

{

    FslIMX25State *s = FSL_IMX25(dev);

    uint8_t i;

    Error *err = NULL;



    object_property_set_bool(OBJECT(&s->cpu), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    object_property_set_bool(OBJECT(&s->avic), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    sysbus_mmio_map(SYS_BUS_DEVICE(&s->avic), 0, FSL_IMX25_AVIC_ADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->avic), 0,

                       qdev_get_gpio_in(DEVICE(&s->cpu), ARM_CPU_IRQ));

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->avic), 1,

                       qdev_get_gpio_in(DEVICE(&s->cpu), ARM_CPU_FIQ));



    object_property_set_bool(OBJECT(&s->ccm), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    sysbus_mmio_map(SYS_BUS_DEVICE(&s->ccm), 0, FSL_IMX25_CCM_ADDR);



    /* Initialize all UARTs */

    for (i = 0; i < FSL_IMX25_NUM_UARTS; i++) {

        static const struct {

            hwaddr addr;

            unsigned int irq;

        } serial_table[FSL_IMX25_NUM_UARTS] = {

            { FSL_IMX25_UART1_ADDR, FSL_IMX25_UART1_IRQ },

            { FSL_IMX25_UART2_ADDR, FSL_IMX25_UART2_IRQ },

            { FSL_IMX25_UART3_ADDR, FSL_IMX25_UART3_IRQ },

            { FSL_IMX25_UART4_ADDR, FSL_IMX25_UART4_IRQ },

            { FSL_IMX25_UART5_ADDR, FSL_IMX25_UART5_IRQ }

        };



        if (i < MAX_SERIAL_PORTS) {

            Chardev *chr;



            chr = serial_hds[i];



            if (!chr) {

                char label[20];

                snprintf(label, sizeof(label), "imx31.uart%d", i);

                chr = qemu_chr_new(label, "null");

            }



            qdev_prop_set_chr(DEVICE(&s->uart[i]), "chardev", chr);

        }



        object_property_set_bool(OBJECT(&s->uart[i]), true, "realized", &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

        sysbus_mmio_map(SYS_BUS_DEVICE(&s->uart[i]), 0, serial_table[i].addr);

        sysbus_connect_irq(SYS_BUS_DEVICE(&s->uart[i]), 0,

                           qdev_get_gpio_in(DEVICE(&s->avic),

                                            serial_table[i].irq));

    }



    /* Initialize all GPT timers */

    for (i = 0; i < FSL_IMX25_NUM_GPTS; i++) {

        static const struct {

            hwaddr addr;

            unsigned int irq;

        } gpt_table[FSL_IMX25_NUM_GPTS] = {

            { FSL_IMX25_GPT1_ADDR, FSL_IMX25_GPT1_IRQ },

            { FSL_IMX25_GPT2_ADDR, FSL_IMX25_GPT2_IRQ },

            { FSL_IMX25_GPT3_ADDR, FSL_IMX25_GPT3_IRQ },

            { FSL_IMX25_GPT4_ADDR, FSL_IMX25_GPT4_IRQ }

        };



        s->gpt[i].ccm = IMX_CCM(&s->ccm);



        object_property_set_bool(OBJECT(&s->gpt[i]), true, "realized", &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

        sysbus_mmio_map(SYS_BUS_DEVICE(&s->gpt[i]), 0, gpt_table[i].addr);

        sysbus_connect_irq(SYS_BUS_DEVICE(&s->gpt[i]), 0,

                           qdev_get_gpio_in(DEVICE(&s->avic),

                                            gpt_table[i].irq));

    }



    /* Initialize all EPIT timers */

    for (i = 0; i < FSL_IMX25_NUM_EPITS; i++) {

        static const struct {

            hwaddr addr;

            unsigned int irq;

        } epit_table[FSL_IMX25_NUM_EPITS] = {

            { FSL_IMX25_EPIT1_ADDR, FSL_IMX25_EPIT1_IRQ },

            { FSL_IMX25_EPIT2_ADDR, FSL_IMX25_EPIT2_IRQ }

        };



        s->epit[i].ccm = IMX_CCM(&s->ccm);



        object_property_set_bool(OBJECT(&s->epit[i]), true, "realized", &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

        sysbus_mmio_map(SYS_BUS_DEVICE(&s->epit[i]), 0, epit_table[i].addr);

        sysbus_connect_irq(SYS_BUS_DEVICE(&s->epit[i]), 0,

                           qdev_get_gpio_in(DEVICE(&s->avic),

                                            epit_table[i].irq));

    }



    qdev_set_nic_properties(DEVICE(&s->fec), &nd_table[0]);



    object_property_set_bool(OBJECT(&s->fec), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    sysbus_mmio_map(SYS_BUS_DEVICE(&s->fec), 0, FSL_IMX25_FEC_ADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->fec), 0,

                       qdev_get_gpio_in(DEVICE(&s->avic), FSL_IMX25_FEC_IRQ));





    /* Initialize all I2C */

    for (i = 0; i < FSL_IMX25_NUM_I2CS; i++) {

        static const struct {

            hwaddr addr;

            unsigned int irq;

        } i2c_table[FSL_IMX25_NUM_I2CS] = {

            { FSL_IMX25_I2C1_ADDR, FSL_IMX25_I2C1_IRQ },

            { FSL_IMX25_I2C2_ADDR, FSL_IMX25_I2C2_IRQ },

            { FSL_IMX25_I2C3_ADDR, FSL_IMX25_I2C3_IRQ }

        };



        object_property_set_bool(OBJECT(&s->i2c[i]), true, "realized", &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

        sysbus_mmio_map(SYS_BUS_DEVICE(&s->i2c[i]), 0, i2c_table[i].addr);

        sysbus_connect_irq(SYS_BUS_DEVICE(&s->i2c[i]), 0,

                           qdev_get_gpio_in(DEVICE(&s->avic),

                                            i2c_table[i].irq));

    }



    /* Initialize all GPIOs */

    for (i = 0; i < FSL_IMX25_NUM_GPIOS; i++) {

        static const struct {

            hwaddr addr;

            unsigned int irq;

        } gpio_table[FSL_IMX25_NUM_GPIOS] = {

            { FSL_IMX25_GPIO1_ADDR, FSL_IMX25_GPIO1_IRQ },

            { FSL_IMX25_GPIO2_ADDR, FSL_IMX25_GPIO2_IRQ },

            { FSL_IMX25_GPIO3_ADDR, FSL_IMX25_GPIO3_IRQ },

            { FSL_IMX25_GPIO4_ADDR, FSL_IMX25_GPIO4_IRQ }

        };



        object_property_set_bool(OBJECT(&s->gpio[i]), true, "realized", &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

        sysbus_mmio_map(SYS_BUS_DEVICE(&s->gpio[i]), 0, gpio_table[i].addr);

        /* Connect GPIO IRQ to PIC */

        sysbus_connect_irq(SYS_BUS_DEVICE(&s->gpio[i]), 0,

                           qdev_get_gpio_in(DEVICE(&s->avic),

                                            gpio_table[i].irq));

    }



    /* initialize 2 x 16 KB ROM */

    memory_region_init_rom_nomigrate(&s->rom[0], NULL,

                           "imx25.rom0", FSL_IMX25_ROM0_SIZE, &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    memory_region_add_subregion(get_system_memory(), FSL_IMX25_ROM0_ADDR,

                                &s->rom[0]);

    memory_region_init_rom_nomigrate(&s->rom[1], NULL,

                           "imx25.rom1", FSL_IMX25_ROM1_SIZE, &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    memory_region_add_subregion(get_system_memory(), FSL_IMX25_ROM1_ADDR,

                                &s->rom[1]);



    /* initialize internal RAM (128 KB) */

    memory_region_init_ram(&s->iram, NULL, "imx25.iram", FSL_IMX25_IRAM_SIZE,

                           &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    memory_region_add_subregion(get_system_memory(), FSL_IMX25_IRAM_ADDR,

                                &s->iram);



    /* internal RAM (128 KB) is aliased over 128 MB - 128 KB */

    memory_region_init_alias(&s->iram_alias, NULL, "imx25.iram_alias",

                             &s->iram, 0, FSL_IMX25_IRAM_ALIAS_SIZE);

    memory_region_add_subregion(get_system_memory(), FSL_IMX25_IRAM_ALIAS_ADDR,

                                &s->iram_alias);

}
