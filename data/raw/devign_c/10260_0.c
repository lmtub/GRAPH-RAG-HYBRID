static int nic_can_receive(NetClientState *nc)

{

    EEPRO100State *s = qemu_get_nic_opaque(nc);

    TRACE(RXTX, logout("%p\n", s));

    return get_ru_state(s) == ru_ready;

#if 0

    return !eepro100_buffer_full(s);

#endif

}
