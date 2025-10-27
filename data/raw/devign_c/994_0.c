void do_info_usernet(Monitor *mon)

{

    SlirpState *s;



    TAILQ_FOREACH(s, &slirp_stacks, entry) {

        monitor_printf(mon, "VLAN %d (%s):\n", s->vc->vlan->id, s->vc->name);

        slirp_connection_info(s->slirp, mon);

    }

}
