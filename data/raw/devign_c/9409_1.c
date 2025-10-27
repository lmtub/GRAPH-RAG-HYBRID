static void fsl_imx31_realize(DeviceState *dev, Error **errp)

{

    FslIMX31State *s = FSL_IMX31(dev);

    uint16_t i;

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

    sysbus_mmio_map(SYS_BUS_DEVICE(&s->avic), 0, FSL_IMX31_AVIC_ADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->avic), 0,

                       qdev_get_gpio_in(DEVICE(&s->cpu), ARM_CPU_IRQ));

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->avic), 1,

                       qdev_get_gpio_in(DEVICE(&s->cpu), ARM_CPU_FIQ));



    object_property_set_bool(OBJECT(&s->ccm), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    sysbus_mmio_map(SYS_BUS_DEVICE(&s->ccm), 0, FSL_IMX31_CCM_ADDR);



    /* Initialize all UARTS */

    for (i = 0; i < FSL_IMX31_NUM_UARTS; i++) {

        static const struct {

            hwaddr addr;

            unsigned int irq;

        } serial_table[FSL_IMX31_NUM_UARTS] = {

            { FSL_IMX31_UART1_ADDR, FSL_IMX31_UART1_IRQ },

            { FSL_IMX31_UART2_ADDR, FSL_IMX31_UART2_IRQ },

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



    s->gpt.ccm = IMX_CCM(&s->ccm);



    object_property_set_bool(OBJECT(&s->gpt), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    sysbus_mmio_map(SYS_BUS_DEVICE(&s->gpt), 0, FSL_IMX31_GPT_ADDR);

    sysbus_connect_irq(SYS_BUS_DEVICE(&s->gpt), 0,

                       qdev_get_gpio_in(DEVICE(&s->avic), FSL_IMX31_GPT_IRQ));



    /* Initialize all EPIT timers */

    for (i = 0; i < FSL_IMX31_NUM_EPITS; i++) {

        static const struct {

            hwaddr addr;

            unsigned int irq;

        } epit_table[FSL_IMX31_NUM_EPITS] = {

            { FSL_IMX31_EPIT1_ADDR, FSL_IMX31_EPIT1_IRQ },

            { FSL_IMX31_EPIT2_ADDR, FSL_IMX31_EPIT2_IRQ },

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



    /* Initialize all I2C */

    for (i = 0; i < FSL_IMX31_NUM_I2CS; i++) {

        static const struct {

            hwaddr addr;

            unsigned int irq;

        } i2c_table[FSL_IMX31_NUM_I2CS] = {

            { FSL_IMX31_I2C1_ADDR, FSL_IMX31_I2C1_IRQ },

            { FSL_IMX31_I2C2_ADDR, FSL_IMX31_I2C2_IRQ },

            { FSL_IMX31_I2C3_ADDR, FSL_IMX31_I2C3_IRQ }

        };



        /* Initialize the I2C */

        object_property_set_bool(OBJECT(&s->i2c[i]), true, "realized", &err);

        if (err) {

            error_propagate(errp, err);

            return;

        }

        /* Map I2C memory */

        sysbus_mmio_map(SYS_BUS_DEVICE(&s->i2c[i]), 0, i2c_table[i].addr);

        /* Connect I2C IRQ to PIC */

        sysbus_connect_irq(SYS_BUS_DEVICE(&s->i2c[i]), 0,

                           qdev_get_gpio_in(DEVICE(&s->avic),

                                            i2c_table[i].irq));

    }



    /* Initialize all GPIOs */

    for (i = 0; i < FSL_IMX31_NUM_GPIOS; i++) {

        static const struct {

            hwaddr addr;

            unsigned int irq;

        } gpio_table[FSL_IMX31_NUM_GPIOS] = {

            { FSL_IMX31_GPIO1_ADDR, FSL_IMX31_GPIO1_IRQ },

            { FSL_IMX31_GPIO2_ADDR, FSL_IMX31_GPIO2_IRQ },

            { FSL_IMX31_GPIO3_ADDR, FSL_IMX31_GPIO3_IRQ }

        };



        object_property_set_bool(OBJECT(&s->gpio[i]), false, "has-edge-sel",

                                 &error_abort);

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



    /* On a real system, the first 16k is a `secure boot rom' */

    memory_region_init_rom_nomigrate(&s->secure_rom, NULL, "imx31.secure_rom",

                           FSL_IMX31_SECURE_ROM_SIZE, &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    memory_region_add_subregion(get_system_memory(), FSL_IMX31_SECURE_ROM_ADDR,

                                &s->secure_rom);



    /* There is also a 16k ROM */

    memory_region_init_rom_nomigrate(&s->rom, NULL, "imx31.rom",

                           FSL_IMX31_ROM_SIZE, &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    memory_region_add_subregion(get_system_memory(), FSL_IMX31_ROM_ADDR,

                                &s->rom);



    /* initialize internal RAM (16 KB) */

    memory_region_init_ram(&s->iram, NULL, "imx31.iram", FSL_IMX31_IRAM_SIZE,

                           &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }

    memory_region_add_subregion(get_system_memory(), FSL_IMX31_IRAM_ADDR,

                                &s->iram);



    /* internal RAM (16 KB) is aliased over 256 MB - 16 KB */

    memory_region_init_alias(&s->iram_alias, NULL, "imx31.iram_alias",

                             &s->iram, 0, FSL_IMX31_IRAM_ALIAS_SIZE);

    memory_region_add_subregion(get_system_memory(), FSL_IMX31_IRAM_ALIAS_ADDR,

                                &s->iram_alias);

}
