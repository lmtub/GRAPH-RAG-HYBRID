static int eth_can_rx(NetClientState *nc)

{

    struct xlx_ethlite *s = DO_UPCAST(NICState, nc, nc)->opaque;

    int r;

    r = !(s->regs[R_RX_CTRL0] & CTRL_S);

    return r;

}
