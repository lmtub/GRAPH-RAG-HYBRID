static int n8x0_atag_setup(void *p, int model)

{

    uint8_t *b;

    uint16_t *w;

    uint32_t *l;

    struct omap_gpiosw_info_s *gpiosw;

    struct omap_partition_info_s *partition;

    const char *tag;



    w = p;



    stw_p(w++, OMAP_TAG_UART);			/* u16 tag */

    stw_p(w++, 4);				/* u16 len */

    stw_p(w++, (1 << 2) | (1 << 1) | (1 << 0)); /* uint enabled_uarts */

    w++;



#if 0

    stw_p(w++, OMAP_TAG_SERIAL_CONSOLE);	/* u16 tag */

    stw_p(w++, 4);				/* u16 len */

    stw_p(w++, XLDR_LL_UART + 1);		/* u8 console_uart */

    stw_p(w++, 115200);				/* u32 console_speed */

#endif



    stw_p(w++, OMAP_TAG_LCD);			/* u16 tag */

    stw_p(w++, 36);				/* u16 len */

    strcpy((void *) w, "QEMU LCD panel");	/* char panel_name[16] */

    w += 8;

    strcpy((void *) w, "blizzard");		/* char ctrl_name[16] */

    w += 8;

    stw_p(w++, N810_BLIZZARD_RESET_GPIO);	/* TODO: n800 s16 nreset_gpio */

    stw_p(w++, 24);				/* u8 data_lines */



    stw_p(w++, OMAP_TAG_CBUS);			/* u16 tag */

    stw_p(w++, 8);				/* u16 len */

    stw_p(w++, N8X0_CBUS_CLK_GPIO);		/* s16 clk_gpio */

    stw_p(w++, N8X0_CBUS_DAT_GPIO);		/* s16 dat_gpio */

    stw_p(w++, N8X0_CBUS_SEL_GPIO);		/* s16 sel_gpio */

    w++;



    stw_p(w++, OMAP_TAG_EM_ASIC_BB5);		/* u16 tag */

    stw_p(w++, 4);				/* u16 len */

    stw_p(w++, N8X0_RETU_GPIO);			/* s16 retu_irq_gpio */

    stw_p(w++, N8X0_TAHVO_GPIO);		/* s16 tahvo_irq_gpio */



    gpiosw = (model == 810) ? n810_gpiosw_info : n800_gpiosw_info;

    for (; gpiosw->name; gpiosw++) {

        stw_p(w++, OMAP_TAG_GPIO_SWITCH);	/* u16 tag */

        stw_p(w++, 20);				/* u16 len */

        strcpy((void *) w, gpiosw->name);	/* char name[12] */

        w += 6;

        stw_p(w++, gpiosw->line);		/* u16 gpio */

        stw_p(w++, gpiosw->type);

        stw_p(w++, 0);

        stw_p(w++, 0);

    }



    stw_p(w++, OMAP_TAG_NOKIA_BT);		/* u16 tag */

    stw_p(w++, 12);				/* u16 len */

    b = (void *) w;

    stb_p(b++, 0x01);				/* u8 chip_type	(CSR) */

    stb_p(b++, N8X0_BT_WKUP_GPIO);		/* u8 bt_wakeup_gpio */

    stb_p(b++, N8X0_BT_HOST_WKUP_GPIO);		/* u8 host_wakeup_gpio */

    stb_p(b++, N8X0_BT_RESET_GPIO);		/* u8 reset_gpio */

    stb_p(b++, BT_UART + 1);			/* u8 bt_uart */

    memcpy(b, &n8x0_bd_addr, 6);		/* u8 bd_addr[6] */

    b += 6;

    stb_p(b++, 0x02);				/* u8 bt_sysclk (38.4) */

    w = (void *) b;



    stw_p(w++, OMAP_TAG_WLAN_CX3110X);		/* u16 tag */

    stw_p(w++, 8);				/* u16 len */

    stw_p(w++, 0x25);				/* u8 chip_type */

    stw_p(w++, N8X0_WLAN_PWR_GPIO);		/* s16 power_gpio */

    stw_p(w++, N8X0_WLAN_IRQ_GPIO);		/* s16 irq_gpio */

    stw_p(w++, -1);				/* s16 spi_cs_gpio */



    stw_p(w++, OMAP_TAG_MMC);			/* u16 tag */

    stw_p(w++, 16);				/* u16 len */

    if (model == 810) {

        stw_p(w++, 0x23f);			/* unsigned flags */

        stw_p(w++, -1);				/* s16 power_pin */

        stw_p(w++, -1);				/* s16 switch_pin */

        stw_p(w++, -1);				/* s16 wp_pin */

        stw_p(w++, 0x240);			/* unsigned flags */

        stw_p(w++, 0xc000);			/* s16 power_pin */

        stw_p(w++, 0x0248);			/* s16 switch_pin */

        stw_p(w++, 0xc000);			/* s16 wp_pin */

    } else {

        stw_p(w++, 0xf);			/* unsigned flags */

        stw_p(w++, -1);				/* s16 power_pin */

        stw_p(w++, -1);				/* s16 switch_pin */

        stw_p(w++, -1);				/* s16 wp_pin */

        stw_p(w++, 0);				/* unsigned flags */

        stw_p(w++, 0);				/* s16 power_pin */

        stw_p(w++, 0);				/* s16 switch_pin */

        stw_p(w++, 0);				/* s16 wp_pin */

    }



    stw_p(w++, OMAP_TAG_TEA5761);		/* u16 tag */

    stw_p(w++, 4);				/* u16 len */

    stw_p(w++, N8X0_TEA5761_CS_GPIO);		/* u16 enable_gpio */

    w++;



    partition = (model == 810) ? n810_part_info : n800_part_info;

    for (; partition->name; partition++) {

        stw_p(w++, OMAP_TAG_PARTITION);		/* u16 tag */

        stw_p(w++, 28);				/* u16 len */

        strcpy((void *) w, partition->name);	/* char name[16] */

        l = (void *) (w + 8);

        stl_p(l++, partition->size);		/* unsigned int size */

        stl_p(l++, partition->offset);		/* unsigned int offset */

        stl_p(l++, partition->mask);		/* unsigned int mask_flags */

        w = (void *) l;

    }



    stw_p(w++, OMAP_TAG_BOOT_REASON);		/* u16 tag */

    stw_p(w++, 12);				/* u16 len */

#if 0

    strcpy((void *) w, "por");			/* char reason_str[12] */

    strcpy((void *) w, "charger");		/* char reason_str[12] */

    strcpy((void *) w, "32wd_to");		/* char reason_str[12] */

    strcpy((void *) w, "sw_rst");		/* char reason_str[12] */

    strcpy((void *) w, "mbus");			/* char reason_str[12] */

    strcpy((void *) w, "unknown");		/* char reason_str[12] */

    strcpy((void *) w, "swdg_to");		/* char reason_str[12] */

    strcpy((void *) w, "sec_vio");		/* char reason_str[12] */

    strcpy((void *) w, "pwr_key");		/* char reason_str[12] */

    strcpy((void *) w, "rtc_alarm");		/* char reason_str[12] */

#else

    strcpy((void *) w, "pwr_key");		/* char reason_str[12] */

#endif

    w += 6;



    tag = (model == 810) ? "RX-44" : "RX-34";

    stw_p(w++, OMAP_TAG_VERSION_STR);		/* u16 tag */

    stw_p(w++, 24);				/* u16 len */

    strcpy((void *) w, "product");		/* char component[12] */

    w += 6;

    strcpy((void *) w, tag);			/* char version[12] */

    w += 6;



    stw_p(w++, OMAP_TAG_VERSION_STR);		/* u16 tag */

    stw_p(w++, 24);				/* u16 len */

    strcpy((void *) w, "hw-build");		/* char component[12] */

    w += 6;

    strcpy((void *) w, "QEMU ");

    pstrcat((void *) w, 12, qemu_get_version()); /* char version[12] */

    w += 6;



    tag = (model == 810) ? "1.1.10-qemu" : "1.1.6-qemu";

    stw_p(w++, OMAP_TAG_VERSION_STR);		/* u16 tag */

    stw_p(w++, 24);				/* u16 len */

    strcpy((void *) w, "nolo");			/* char component[12] */

    w += 6;

    strcpy((void *) w, tag);			/* char version[12] */

    w += 6;



    return (void *) w - p;

}
