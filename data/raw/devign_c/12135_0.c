static void isa_ne2000_cleanup(NetClientState *nc)

{

    NE2000State *s = qemu_get_nic_opaque(nc);



    s->nic = NULL;

}
