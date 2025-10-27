static void qemu_announce_self_once(void *opaque)

{

    int i, len;

    VLANState *vlan;

    VLANClientState *vc;

    uint8_t buf[256];

    static int count = SELF_ANNOUNCE_ROUNDS;

    QEMUTimer *timer = *(QEMUTimer **)opaque;



    for (i = 0; i < MAX_NICS; i++) {

        if (!nd_table[i].used)

            continue;

        len = announce_self_create(buf, nd_table[i].macaddr);

        vlan = nd_table[i].vlan;

        QTAILQ_FOREACH(vc, &vlan->clients, next) {

            vc->receive(vc, buf, len);

        }

    }

    if (count--) {

	    qemu_mod_timer(timer, qemu_get_clock(rt_clock) + 100);

    } else {

	    qemu_del_timer(timer);

	    qemu_free_timer(timer);

    }

}
