e1000_cleanup(NetClientState *nc)

{

    E1000State *s = qemu_get_nic_opaque(nc);



    s->nic = NULL;

}
