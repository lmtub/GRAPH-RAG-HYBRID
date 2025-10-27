static void minimac2_cleanup(NetClientState *nc)

{

    MilkymistMinimac2State *s = qemu_get_nic_opaque(nc);



    s->nic = NULL;

}
