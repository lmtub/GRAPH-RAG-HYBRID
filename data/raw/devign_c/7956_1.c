static int imx_eth_can_receive(NetClientState *nc)

{

    IMXFECState *s = IMX_FEC(qemu_get_nic_opaque(nc));



    FEC_PRINTF("\n");



    return s->regs[ENET_RDAR] ? 1 : 0;

}
