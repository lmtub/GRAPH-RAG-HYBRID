static void imx25_pdk_init(MachineState *machine)

{

    IMX25PDK *s = g_new0(IMX25PDK, 1);

    unsigned int ram_size;

    unsigned int alias_offset;

    int i;



    object_initialize(&s->soc, sizeof(s->soc), TYPE_FSL_IMX25);

    object_property_add_child(OBJECT(machine), "soc", OBJECT(&s->soc),

                              &error_abort);



    object_property_set_bool(OBJECT(&s->soc), true, "realized", &error_fatal);



    /* We need to initialize our memory */

    if (machine->ram_size > (FSL_IMX25_SDRAM0_SIZE + FSL_IMX25_SDRAM1_SIZE)) {

        error_report("WARNING: RAM size " RAM_ADDR_FMT " above max supported, "

                     "reduced to %x", machine->ram_size,

                     FSL_IMX25_SDRAM0_SIZE + FSL_IMX25_SDRAM1_SIZE);

        machine->ram_size = FSL_IMX25_SDRAM0_SIZE + FSL_IMX25_SDRAM1_SIZE;

    }



    memory_region_allocate_system_memory(&s->ram, NULL, "imx25.ram",

                                         machine->ram_size);

    memory_region_add_subregion(get_system_memory(), FSL_IMX25_SDRAM0_ADDR,

                                &s->ram);



    /* initialize the alias memory if any */

    for (i = 0, ram_size = machine->ram_size, alias_offset = 0;

         (i < 2) && ram_size; i++) {

        unsigned int size;

        static const struct {

            hwaddr addr;

            unsigned int size;

        } ram[2] = {

            { FSL_IMX25_SDRAM0_ADDR, FSL_IMX25_SDRAM0_SIZE },

            { FSL_IMX25_SDRAM1_ADDR, FSL_IMX25_SDRAM1_SIZE },

        };



        size = MIN(ram_size, ram[i].size);



        ram_size -= size;



        if (size < ram[i].size) {

            memory_region_init_alias(&s->ram_alias, NULL, "ram.alias",

                                     &s->ram, alias_offset, ram[i].size - size);

            memory_region_add_subregion(get_system_memory(),

                                        ram[i].addr + size, &s->ram_alias);

        }



        alias_offset += ram[i].size;

    }



    imx25_pdk_binfo.ram_size = machine->ram_size;

    imx25_pdk_binfo.kernel_filename = machine->kernel_filename;

    imx25_pdk_binfo.kernel_cmdline = machine->kernel_cmdline;

    imx25_pdk_binfo.initrd_filename = machine->initrd_filename;

    imx25_pdk_binfo.loader_start = FSL_IMX25_SDRAM0_ADDR;

    imx25_pdk_binfo.board_id = 1771,

    imx25_pdk_binfo.nb_cpus = 1;



    /*

     * We test explicitly for qtest here as it is not done (yet?) in

     * arm_load_kernel(). Without this the "make check" command would

     * fail.

     */

    if (!qtest_enabled()) {

        arm_load_kernel(&s->soc.cpu, &imx25_pdk_binfo);

    } else {

        /*

         * This I2C device doesn't exist on the real board.

         * We add it here (only on qtest usage) to be able to do a bit

         * of simple qtest. See "make check" for details.

         */

        i2c_create_slave((I2CBus *)qdev_get_child_bus(DEVICE(&s->soc.i2c[0]),

                                                      "i2c-bus.0"),

                         "ds1338", 0x68);

    }

}
