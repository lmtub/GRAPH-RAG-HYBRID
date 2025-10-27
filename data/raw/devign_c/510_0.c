static void pl061_register_devices(void)

{

    sysbus_register_dev("pl061", sizeof(pl061_state),

                        pl061_init_arm);

    sysbus_register_dev("pl061_luminary", sizeof(pl061_state),

                        pl061_init_luminary);

}
