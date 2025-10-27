static void stellaris_enet_cleanup(NetClientState *nc)

{

    stellaris_enet_state *s = qemu_get_nic_opaque(nc);



    s->nic = NULL;

}
