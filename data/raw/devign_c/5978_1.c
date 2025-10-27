e1000_can_receive(VLANClientState *nc)

{

    E1000State *s = DO_UPCAST(NICState, nc, nc)->opaque;



    return (s->mac_reg[RCTL] & E1000_RCTL_EN);

}
