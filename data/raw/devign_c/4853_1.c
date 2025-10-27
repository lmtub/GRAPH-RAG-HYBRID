static void omap_i2c_send(I2CAdapter *i2c, uint8_t addr,

                          const uint8_t *buf, uint16_t len)

{

    OMAPI2C *s = (OMAPI2C *)i2c;

    uint16_t data;



    omap_i2c_set_slave_addr(s, addr);



    data = len;

    memwrite(s->addr + OMAP_I2C_CNT, &data, 2);



    data = OMAP_I2C_CON_I2C_EN |

           OMAP_I2C_CON_TRX |

           OMAP_I2C_CON_MST |

           OMAP_I2C_CON_STT |

           OMAP_I2C_CON_STP;

    memwrite(s->addr + OMAP_I2C_CON, &data, 2);

    memread(s->addr + OMAP_I2C_CON, &data, 2);

    g_assert((data & OMAP_I2C_CON_STP) != 0);



    memread(s->addr + OMAP_I2C_STAT, &data, 2);

    g_assert((data & OMAP_I2C_STAT_NACK) == 0);



    while (len > 1) {

        memread(s->addr + OMAP_I2C_STAT, &data, 2);

        g_assert((data & OMAP_I2C_STAT_XRDY) != 0);



        memwrite(s->addr + OMAP_I2C_DATA, buf, 2);

        buf = (uint8_t *)buf + 2;

        len -= 2;

    }

    if (len == 1) {

        memread(s->addr + OMAP_I2C_STAT, &data, 2);

        g_assert((data & OMAP_I2C_STAT_XRDY) != 0);



        memwrite(s->addr + OMAP_I2C_DATA, buf, 1);

    }



    memread(s->addr + OMAP_I2C_CON, &data, 2);

    g_assert((data & OMAP_I2C_CON_STP) == 0);

}
