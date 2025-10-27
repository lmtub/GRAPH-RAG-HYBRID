static void aw_emac_cleanup(NetClientState *nc)

{

    AwEmacState *s = qemu_get_nic_opaque(nc);



    s->nic = NULL;

}
