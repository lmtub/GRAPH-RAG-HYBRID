static void tap_set_sndbuf(TAPState *s, int sndbuf, Monitor *mon)

{

#ifdef TUNSETSNDBUF

    if (ioctl(s->fd, TUNSETSNDBUF, &sndbuf) == -1) {

        config_error(mon, "TUNSETSNDBUF ioctl failed: %s\n",

                     strerror(errno));

    }

#else

    config_error(mon, "No '-net tap,sndbuf=<nbytes>' support available\n");

#endif

}
