static void lan9118_cleanup(NetClientState *nc)

{

    lan9118_state *s = qemu_get_nic_opaque(nc);



    s->nic = NULL;

}
