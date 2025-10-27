static int e1000_post_load(void *opaque, int version_id)

{

    E1000State *s = opaque;

    NetClientState *nc = qemu_get_queue(s->nic);



    if (!(s->compat_flags & E1000_FLAG_MIT)) {

        s->mac_reg[ITR] = s->mac_reg[RDTR] = s->mac_reg[RADV] =

            s->mac_reg[TADV] = 0;

        s->mit_irq_level = false;

    }

    s->mit_ide = 0;

    s->mit_timer_on = false;



    /* nc.link_down can't be migrated, so infer link_down according

     * to link status bit in mac_reg[STATUS].

     * Alternatively, restart link negotiation if it was in progress. */

    nc->link_down = (s->mac_reg[STATUS] & E1000_STATUS_LU) == 0;



    if (s->compat_flags & E1000_FLAG_AUTONEG &&

        s->phy_reg[PHY_CTRL] & MII_CR_AUTO_NEG_EN &&

        s->phy_reg[PHY_CTRL] & MII_CR_RESTART_AUTO_NEG &&

        !(s->phy_reg[PHY_STATUS] & MII_SR_AUTONEG_COMPLETE)) {

        nc->link_down = false;

        timer_mod(s->autoneg_timer, qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + 500);

    }



    return 0;

}
