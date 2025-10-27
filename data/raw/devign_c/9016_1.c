set_phy_ctrl(E1000State *s, int index, uint16_t val)

{

    if ((val & MII_CR_AUTO_NEG_EN) && (val & MII_CR_RESTART_AUTO_NEG)) {

        qemu_get_queue(s->nic)->link_down = true;

        e1000_link_down(s);

        s->phy_reg[PHY_STATUS] &= ~MII_SR_AUTONEG_COMPLETE;

        DBGOUT(PHY, "Start link auto negotiation\n");

        qemu_mod_timer(s->autoneg_timer, qemu_get_clock_ms(vm_clock) + 500);

    }

}
