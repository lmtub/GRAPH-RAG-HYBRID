static void musicpal_register_devices(void)

{

    sysbus_register_dev("mv88w8618_pic", sizeof(mv88w8618_pic_state),

                        mv88w8618_pic_init);

    sysbus_register_dev("mv88w8618_pit", sizeof(mv88w8618_pit_state),

                        mv88w8618_pit_init);

    sysbus_register_dev("mv88w8618_flashcfg", sizeof(mv88w8618_flashcfg_state),

                        mv88w8618_flashcfg_init);

    sysbus_register_dev("mv88w8618_eth", sizeof(mv88w8618_eth_state),

                        mv88w8618_eth_init);

    sysbus_register_dev("mv88w8618_wlan", sizeof(SysBusDevice),

                        mv88w8618_wlan_init);

    sysbus_register_dev("musicpal_lcd", sizeof(musicpal_lcd_state),

                        musicpal_lcd_init);

    sysbus_register_withprop(&musicpal_gpio_info);

    sysbus_register_dev("musicpal_key", sizeof(musicpal_key_state),

                        musicpal_key_init);

}
