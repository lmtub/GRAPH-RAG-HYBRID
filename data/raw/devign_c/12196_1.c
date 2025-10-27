e1000_link_up(E1000State *s)
{
    s->mac_reg[STATUS] |= E1000_STATUS_LU;
    s->phy_reg[PHY_STATUS] |= MII_SR_LINK_STATUS;
}