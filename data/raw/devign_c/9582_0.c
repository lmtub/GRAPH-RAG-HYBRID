e1000_set_link_status(NetClientState *nc)

{

    E1000State *s = qemu_get_nic_opaque(nc);

    uint32_t old_status = s->mac_reg[STATUS];



    if (nc->link_down) {

        e1000_link_down(s);

    } else {

        if (s->compat_flags & E1000_FLAG_AUTONEG &&

            s->phy_reg[PHY_CTRL] & MII_CR_AUTO_NEG_EN &&

            s->phy_reg[PHY_CTRL] & MII_CR_RESTART_AUTO_NEG &&

            !(s->phy_reg[PHY_STATUS] & MII_SR_AUTONEG_COMPLETE)) {

            /* emulate auto-negotiation if supported */

            timer_mod(s->autoneg_timer,

                      qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + 500);

        } else {

            e1000_link_up(s);

        }

    }



    if (s->mac_reg[STATUS] != old_status)

        set_ics(s, 0, E1000_ICR_LSC);

}
