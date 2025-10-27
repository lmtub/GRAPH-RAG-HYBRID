static void e1000_reset(void *opaque)

{

    E1000State *d = opaque;





    qemu_del_timer(d->autoneg_timer);

    memset(d->phy_reg, 0, sizeof d->phy_reg);

    memmove(d->phy_reg, phy_reg_init, sizeof phy_reg_init);

    memset(d->mac_reg, 0, sizeof d->mac_reg);

    memmove(d->mac_reg, mac_reg_init, sizeof mac_reg_init);

    d->rxbuf_min_shift = 1;

    memset(&d->tx, 0, sizeof d->tx);



    if (d->nic->nc.link_down) {

        e1000_link_down(d);

    }



    /* Some guests expect pre-initialized RAH/RAL (AddrValid flag + MACaddr) */

    d->mac_reg[RA] = 0;

    d->mac_reg[RA + 1] = E1000_RAH_AV;

    for (i = 0; i < 4; i++) {

        d->mac_reg[RA] |= macaddr[i] << (8 * i);

        d->mac_reg[RA + 1] |= (i < 2) ? macaddr[i + 4] << (8 * i) : 0;

    }

}