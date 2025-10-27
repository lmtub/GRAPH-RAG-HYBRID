set_phy_ctrl(E1000State *s, int index, uint16_t val)

{

    /*

     * QEMU 1.3 does not support link auto-negotiation emulation, so if we

     * migrate during auto negotiation, after migration the link will be

     * down.

     */

    if (!(s->compat_flags & E1000_FLAG_AUTONEG)) {

        return;

    }

    if ((val & MII_CR_AUTO_NEG_EN) && (val & MII_CR_RESTART_AUTO_NEG)) {

        e1000_link_down(s);

        DBGOUT(PHY, "Start link auto negotiation\n");

        timer_mod(s->autoneg_timer, qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + 500);

    }

}
