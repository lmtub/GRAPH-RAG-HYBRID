static void mipsnet_cleanup(NetClientState *nc)

{

    MIPSnetState *s = qemu_get_nic_opaque(nc);



    s->nic = NULL;

}
