void omap_mcbsp_i2s_attach(struct omap_mcbsp_s *s, I2SCodec *slave)

{

    s->codec = slave;

    slave->rx_swallow = qemu_allocate_irqs(omap_mcbsp_i2s_swallow, s, 1)[0];

    slave->tx_start = qemu_allocate_irqs(omap_mcbsp_i2s_start, s, 1)[0];

}
